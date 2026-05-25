// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

//! `VsockPacket` provides a thin wrapper over the buffers exchanged via virtio queues.
//! There are two components to a vsock packet, each described by a virtio descriptor chain:
//! - the packet header; and
//! - the packet data/buffer.
//!
//! There is a 1:1 relation between descriptor chains and packets: the first (chain head) holds
//! the header, and the remaining descriptors (if any) hold the data. The data descriptors are
//! only present for data packets (VSOCK_OP_RW).
//!
//! `VsockPacket` wraps these two buffers and provides direct access to the data stored
//! in guest memory. This is done to avoid unnecessarily copying data from guest memory
//! to temporary buffers, before passing it on to the vsock backend.

use std::ops::Deref;

use byteorder::{ByteOrder, LittleEndian};
use virtio_queue::DescriptorChain;
use vm_memory::{Address, Bytes, GuestAddress, GuestMemory};
use vm_virtio::{AccessPlatform, Translatable};

use super::{Result, VsockError, defs};

// The vsock packet header is defined by the C struct:
//
// ```C
// struct virtio_vsock_hdr {
//     le64 src_cid;
//     le64 dst_cid;
//     le32 src_port;
//     le32 dst_port;
//     le32 len;
//     le16 type;
//     le16 op;
//     le32 flags;
//     le32 buf_alloc;
//     le32 fwd_cnt;
// };
// ```
//
// This struct will occupy the buffer pointed to by the head descriptor. We'll be accessing it
// as a byte slice. To that end, we define below the offsets for each field struct, as well as the
// packed struct size, as a bunch of `usize` consts.
// Note that these offsets are only used privately by the `VsockPacket` struct, the public interface
// consisting of getter and setter methods, for each struct field, that will also handle the correct
// endianness.

/// The vsock packet header struct size (when packed).
pub const VSOCK_PKT_HDR_SIZE: usize = 44;

// Source CID.
const HDROFF_SRC_CID: usize = 0;

// Destination CID.
const HDROFF_DST_CID: usize = 8;

// Source port.
const HDROFF_SRC_PORT: usize = 16;

// Destination port.
const HDROFF_DST_PORT: usize = 20;

// Data length (in bytes) - may be 0, if there is no data buffer.
const HDROFF_LEN: usize = 24;

// Socket type. Currently, only connection-oriented streams are defined by the vsock protocol.
const HDROFF_TYPE: usize = 28;

// Operation ID - one of the VSOCK_OP_* values; e.g.
// - VSOCK_OP_RW: a data packet;
// - VSOCK_OP_REQUEST: connection request;
// - VSOCK_OP_RST: forceful connection termination;
// etc (see `super::defs::uapi` for the full list).
const HDROFF_OP: usize = 30;

// Additional options (flags) associated with the current operation (`op`).
// Currently, only used with shutdown requests (VSOCK_OP_SHUTDOWN).
const HDROFF_FLAGS: usize = 32;

// Size (in bytes) of the packet sender receive buffer (for the connection to which this packet
// belongs).
const HDROFF_BUF_ALLOC: usize = 36;

// Number of bytes the sender has received and consumed (for the connection to which this packet
// belongs). For instance, for our Unix backend, this counter would be the total number of bytes
// we have successfully written to a backing Unix socket.
const HDROFF_FWD_CNT: usize = 40;

/// The vsock packet, implemented as a wrapper over a virtq descriptor chain:
/// - the chain head, holding the packet header; and
/// - (optional) buffer, only present for data packets (VSOCK_OP_RW).
///
/// Packet header and data buffer always live in host-owned storage. TX packets
/// copy data out of guest memory at construction time. RX packets fill a
/// host-owned bounce buffer that is later written back to guest memory via
/// [`VsockPacket::commit_buf`]. This keeps `VsockPacket` self-contained: it
/// holds no raw pointers into guest memory and does not need the originating
/// descriptor chain or memory snapshot to remain alive after construction.
pub struct VsockPacket {
    // We still hold the header address in guest memory. We need to write back the modified
    // header in RX buffers.
    guest_hdr_addr: GuestAddress,
    hdr: [u8; VSOCK_PKT_HDR_SIZE],
    buf: Option<Box<[u8]>>,
    /// For RX packets, the guest address that `buf` should be written back to
    /// in [`VsockPacket::commit_buf`]. `None` for TX packets (data flows
    /// guest → host) and for RX packets with no data area.
    guest_buf_addr: Option<GuestAddress>,
}

impl VsockPacket {
    /// Create the packet wrapper from a TX virtq chain head.
    ///
    /// The chain head is expected to hold valid packet header data. A following packet buffer
    /// descriptor can optionally end the chain. Bounds and pointer checks are performed when
    /// creating the wrapper.
    ///
    pub fn from_tx_virtq_head<M>(
        desc_chain: &mut DescriptorChain<M>,
        access_platform: Option<&dyn AccessPlatform>,
    ) -> Result<Self>
    where
        M: Clone + Deref,
        M::Target: GuestMemory,
    {
        let head = desc_chain.next().ok_or(VsockError::HdrDescMissing)?;

        // All buffers in the TX queue must be readable.
        //
        if head.is_write_only() {
            return Err(VsockError::UnreadableDescriptor);
        }

        // The packet header should fit inside the head descriptor.
        if head.len() < VSOCK_PKT_HDR_SIZE as u32 {
            return Err(VsockError::HdrDescTooSmall(head.len()));
        }

        let guest_hdr_addr = head
            .addr()
            .translate_gva(access_platform, VSOCK_PKT_HDR_SIZE)
            .map_err(|_| VsockError::GuestMemory)?;

        // To avoid TOCTOU issues when reading/writing the VSock packet header in guest memory,
        // we need to copy the content of the header in the VMM's memory.
        // After the copy, the hdr content can be trusted since the guest can't change its
        // content anymore.
        let mut hdr = [0u8; VSOCK_PKT_HDR_SIZE];
        desc_chain
            .memory()
            .read_slice(hdr.as_mut_slice(), guest_hdr_addr)
            .map_err(|_| VsockError::GuestMemory)?;

        let mut pkt = Self {
            guest_hdr_addr,
            hdr,
            buf: None,
            guest_buf_addr: None,
        };

        // No point looking for a data/buffer descriptor, if the packet is zero-length.
        if pkt.is_empty() {
            return Ok(pkt);
        }

        // Reject weirdly-sized packets.
        //
        if pkt.len() > defs::MAX_PKT_BUF_SIZE as u32 {
            return Err(VsockError::InvalidPktLen(pkt.len()));
        }

        // Copy the packet data into a host-owned buffer. The data may live in
        // the trailing bytes of the head descriptor or in one or more
        // subsequent descriptors.
        let total_len = pkt.len() as usize;
        let mut owned = vec![0u8; total_len];

        if head.has_next() {
            // Data lives in one or more subsequent descriptors.
            let mut offset = 0usize;
            let mut cur_desc = Some(desc_chain.next().ok_or(VsockError::BufDescMissing)?);

            while let Some(desc) = cur_desc {
                if desc.is_write_only() {
                    return Err(VsockError::UnreadableDescriptor);
                }

                let desc_len = desc.len() as usize;
                if desc_len > 0 && offset < total_len {
                    let to_copy = std::cmp::min(desc_len, total_len - offset);
                    let desc_addr = desc
                        .addr()
                        .translate_gva(access_platform, desc_len)
                        .map_err(|_| VsockError::GuestMemory)?;
                    desc_chain
                        .memory()
                        .read_slice(&mut owned[offset..offset + to_copy], desc_addr)
                        .map_err(|_| VsockError::GuestMemory)?;
                    offset += to_copy;
                }

                cur_desc = if desc.has_next() {
                    Some(desc_chain.next().ok_or(VsockError::BufDescMissing)?)
                } else {
                    None
                };
            }

            if offset < total_len {
                return Err(VsockError::BufDescTooSmall);
            }
        } else {
            // The packet data follows the header in the head descriptor.
            let buf_size_in_head = head.len() as usize - VSOCK_PKT_HDR_SIZE;
            if buf_size_in_head < total_len {
                return Err(VsockError::BufDescTooSmall);
            }
            let buf_addr = head
                .addr()
                .checked_add(VSOCK_PKT_HDR_SIZE as u64)
                .ok_or(VsockError::GuestMemory)?
                .translate_gva(access_platform, total_len)
                .map_err(|_| VsockError::GuestMemory)?;
            desc_chain
                .memory()
                .read_slice(&mut owned[..], buf_addr)
                .map_err(|_| VsockError::GuestMemory)?;
        }

        pkt.buf = Some(owned.into_boxed_slice());
        Ok(pkt)
    }

    /// Create the packet wrapper from an RX virtq chain head.
    ///
    /// There must be two descriptors in the chain, both writable: a header descriptor and a data
    /// descriptor. Bounds and pointer checks are performed when creating the wrapper.
    ///
    pub fn from_rx_virtq_head<M>(
        desc_chain: &mut DescriptorChain<M>,
        access_platform: Option<&dyn AccessPlatform>,
    ) -> Result<Self>
    where
        M: Clone + Deref,
        M::Target: GuestMemory,
    {
        let head = desc_chain.next().ok_or(VsockError::HdrDescMissing)?;

        // All RX buffers must be writable.
        //
        if !head.is_write_only() {
            return Err(VsockError::UnwritableDescriptor);
        }

        // The packet header should fit inside the head descriptor.
        if head.len() < VSOCK_PKT_HDR_SIZE as u32 {
            return Err(VsockError::HdrDescTooSmall(head.len()));
        }

        let guest_hdr_addr = head
            .addr()
            .translate_gva(access_platform, VSOCK_PKT_HDR_SIZE)
            .map_err(|_| VsockError::GuestMemory)?;

        // To avoid TOCTOU issues when reading/writing the VSock packet header in guest memory,
        // we need to copy the content of the header in the VMM's memory.
        // After the copy, the hdr content can be trusted since the guest can't change its
        // content anymore.
        let mut hdr = [0u8; VSOCK_PKT_HDR_SIZE];
        desc_chain
            .memory()
            .read_slice(hdr.as_mut_slice(), guest_hdr_addr)
            .map_err(|_| VsockError::GuestMemory)?;

        // Prior to Linux v6.3 the chain has two descriptors (header + data); newer kernels may
        // pack header and data into a single writable descriptor.
        let (buf_size, buf_addr) = if head.has_next() {
            let buf_desc = desc_chain.next().ok_or(VsockError::BufDescMissing)?;
            let buf_size = buf_desc.len() as usize;

            // TODO: support multi-descriptor RX packets as well, like we do for TX. The
            // commit_buf() machinery would make this straightforward.
            if buf_desc.has_next() {
                return Err(VsockError::BufDescTooSmall);
            }

            if buf_size == 0 {
                (0, None)
            } else {
                let addr = buf_desc
                    .addr()
                    .translate_gva(access_platform, buf_size)
                    .map_err(|_| VsockError::GuestMemory)?;
                (buf_size, Some(addr))
            }
        } else {
            let buf_size: usize = head.len() as usize - VSOCK_PKT_HDR_SIZE;
            if buf_size == 0 {
                (0, None)
            } else {
                let addr = head
                    .addr()
                    .checked_add(VSOCK_PKT_HDR_SIZE as u64)
                    .ok_or(VsockError::GuestMemory)?
                    .translate_gva(access_platform, buf_size)
                    .map_err(|_| VsockError::GuestMemory)?;
                (buf_size, Some(addr))
            }
        };

        let (buf, guest_buf_addr) = if let Some(addr) = buf_addr {
            // Validate up-front that the guest range is mapped, so commit_buf can rely on the
            // write_slice succeeding for well-formed completions.
            desc_chain
                .memory()
                .get_slice(addr, buf_size)
                .map_err(|_| VsockError::GuestMemory)?;
            (Some(vec![0u8; buf_size].into_boxed_slice()), Some(addr))
        } else {
            (None, None)
        };

        Ok(Self {
            guest_hdr_addr,
            hdr,
            buf,
            guest_buf_addr,
        })
    }

    /// Provides in-place, byte-slice, access to the vsock packet header.
    ///
    pub fn hdr(&self) -> &[u8] {
        self.hdr.as_slice()
    }

    /// Provides in-place, byte-slice, mutable access to the vsock packet header.
    ///
    pub fn hdr_mut(&mut self) -> &mut [u8] {
        self.hdr.as_mut_slice()
    }

    /// Writes the local copy of the packet header to the guest memory.
    ///
    pub fn commit_hdr<M: GuestMemory>(&mut self, guest_mem: &M) -> Result<()> {
        if self.len() as usize > defs::MAX_PKT_BUF_SIZE {
            return Err(VsockError::InvalidPktLen(self.len()));
        }

        guest_mem
            .write(self.hdr(), self.guest_hdr_addr)
            .map_err(|_| VsockError::GuestMemory)?;

        Ok(())
    }

    /// Writes the host-owned packet data buffer back to guest memory.
    ///
    /// No-op for TX packets (data flows guest → host and was copied at packet construction)
    /// and for RX packets that have no data to deliver (e.g. control packets that only carry
    /// a header).
    pub fn commit_buf<M: GuestMemory>(&self, guest_mem: &M) -> Result<()> {
        let len = self.len() as usize;
        if len > defs::MAX_PKT_BUF_SIZE {
            return Err(VsockError::InvalidPktLen(self.len()));
        }

        let (Some(buf), Some(addr)) = (self.buf.as_ref(), self.guest_buf_addr) else {
            return Ok(());
        };

        if len > buf.len() {
            return Err(VsockError::InvalidPktLen(self.len()));
        }
        if len == 0 {
            return Ok(());
        }

        guest_mem
            .write_slice(&buf[..len], addr)
            .map_err(|_| VsockError::GuestMemory)?;

        Ok(())
    }

    /// Provides in-place, byte-slice access to the vsock packet data buffer.
    ///
    /// Note: control packets (e.g. connection request or reset) have no data buffer associated.
    ///       For those packets, this method will return `None`.
    /// Also note: for TX packets, `buf` is sized to the packet length (the data copied out of
    ///            guest memory at construction). For RX packets, `buf` is sized to the guest
    ///            descriptor capacity; the actual packet length is set later via [`set_len`]
    ///            and stored in the packet header.
    pub fn buf(&self) -> Option<&[u8]> {
        self.buf.as_deref()
    }

    /// Provides in-place, byte-slice, mutable access to the vsock packet data buffer.
    ///
    /// See [`buf`] for sizing semantics.
    pub fn buf_mut(&mut self) -> Option<&mut [u8]> {
        self.buf.as_deref_mut()
    }

    pub fn src_cid(&self) -> u64 {
        LittleEndian::read_u64(&self.hdr()[HDROFF_SRC_CID..])
    }

    pub fn set_src_cid(&mut self, cid: u64) -> &mut Self {
        LittleEndian::write_u64(&mut self.hdr_mut()[HDROFF_SRC_CID..], cid);
        self
    }

    pub fn dst_cid(&self) -> u64 {
        LittleEndian::read_u64(&self.hdr()[HDROFF_DST_CID..])
    }

    pub fn set_dst_cid(&mut self, cid: u64) -> &mut Self {
        LittleEndian::write_u64(&mut self.hdr_mut()[HDROFF_DST_CID..], cid);
        self
    }

    pub fn src_port(&self) -> u32 {
        LittleEndian::read_u32(&self.hdr()[HDROFF_SRC_PORT..])
    }

    pub fn set_src_port(&mut self, port: u32) -> &mut Self {
        LittleEndian::write_u32(&mut self.hdr_mut()[HDROFF_SRC_PORT..], port);
        self
    }

    pub fn dst_port(&self) -> u32 {
        LittleEndian::read_u32(&self.hdr()[HDROFF_DST_PORT..])
    }

    pub fn set_dst_port(&mut self, port: u32) -> &mut Self {
        LittleEndian::write_u32(&mut self.hdr_mut()[HDROFF_DST_PORT..], port);
        self
    }

    pub fn len(&self) -> u32 {
        LittleEndian::read_u32(&self.hdr()[HDROFF_LEN..])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn set_len(&mut self, len: u32) -> &mut Self {
        LittleEndian::write_u32(&mut self.hdr_mut()[HDROFF_LEN..], len);
        self
    }

    pub fn type_(&self) -> u16 {
        LittleEndian::read_u16(&self.hdr()[HDROFF_TYPE..])
    }

    pub fn set_type(&mut self, type_: u16) -> &mut Self {
        LittleEndian::write_u16(&mut self.hdr_mut()[HDROFF_TYPE..], type_);
        self
    }

    pub fn op(&self) -> u16 {
        LittleEndian::read_u16(&self.hdr()[HDROFF_OP..])
    }

    pub fn set_op(&mut self, op: u16) -> &mut Self {
        LittleEndian::write_u16(&mut self.hdr_mut()[HDROFF_OP..], op);
        self
    }

    pub fn flags(&self) -> u32 {
        LittleEndian::read_u32(&self.hdr()[HDROFF_FLAGS..])
    }

    pub fn set_flags(&mut self, flags: u32) -> &mut Self {
        LittleEndian::write_u32(&mut self.hdr_mut()[HDROFF_FLAGS..], flags);
        self
    }

    pub fn set_flag(&mut self, flag: u32) -> &mut Self {
        self.set_flags(self.flags() | flag);
        self
    }

    pub fn buf_alloc(&self) -> u32 {
        LittleEndian::read_u32(&self.hdr()[HDROFF_BUF_ALLOC..])
    }

    pub fn set_buf_alloc(&mut self, buf_alloc: u32) -> &mut Self {
        LittleEndian::write_u32(&mut self.hdr_mut()[HDROFF_BUF_ALLOC..], buf_alloc);
        self
    }

    pub fn fwd_cnt(&self) -> u32 {
        LittleEndian::read_u32(&self.hdr()[HDROFF_FWD_CNT..])
    }

    pub fn set_fwd_cnt(&mut self, fwd_cnt: u32) -> &mut Self {
        LittleEndian::write_u32(&mut self.hdr_mut()[HDROFF_FWD_CNT..], fwd_cnt);
        self
    }
}

#[cfg(test)]
mod unit_tests {
    use virtio_bindings::virtio_ring::VRING_DESC_F_WRITE;
    use virtio_queue::QueueOwnedT;
    use vm_memory::GuestAddress;
    use vm_virtio::queue::testing::{VirtQueue as GuestQ, VirtqDesc as GuestQDesc};

    use super::super::unit_tests::TestContext;
    use super::*;
    use crate::GuestMemoryMmap;
    use crate::vsock::defs::MAX_PKT_BUF_SIZE;

    macro_rules! create_context {
        ($test_ctx:ident, $handler_ctx:ident) => {
            let $test_ctx = TestContext::new();
            let mut $handler_ctx = $test_ctx.create_epoll_handler_context();
            // For TX packets, hdr.len should be set to a valid value.
            set_pkt_len(1024, &$handler_ctx.guest_txvq.dtable[0], &$test_ctx.mem);
        };
    }

    macro_rules! expect_asm_error {
        (tx, $test_ctx:expr, $handler_ctx:expr, $err:pat) => {
            expect_asm_error!($test_ctx, $handler_ctx, $err, from_tx_virtq_head, 1);
        };
        (rx, $test_ctx:expr, $handler_ctx:expr, $err:pat) => {
            expect_asm_error!($test_ctx, $handler_ctx, $err, from_rx_virtq_head, 0);
        };
        ($test_ctx:expr, $handler_ctx:expr, $err:pat, $ctor:ident, $vq:expr) => {
            match VsockPacket::$ctor(
                &mut $handler_ctx.handler.queues[$vq]
                    .iter(&$test_ctx.mem)
                    .unwrap()
                    .next()
                    .unwrap(),
                None,
            ) {
                Err($err) => (),
                Ok(_) => panic!("Packet assembly should've failed!"),
                Err(other) => panic!("Packet assembly failed with: {:?}", other),
            }
        };
    }

    fn set_pkt_len(len: u32, guest_desc: &GuestQDesc, mem: &GuestMemoryMmap) {
        let hdr_gpa = guest_desc.addr.get();
        mem.write_slice(
            &len.to_le_bytes(),
            GuestAddress(hdr_gpa + HDROFF_LEN as u64),
        )
        .unwrap();
    }

    #[test]
    fn test_tx_packet_assembly() {
        // Test case: successful TX packet assembly.
        {
            create_context!(test_ctx, handler_ctx);

            let pkt = VsockPacket::from_tx_virtq_head(
                &mut handler_ctx.handler.queues[1]
                    .iter(&test_ctx.mem)
                    .unwrap()
                    .next()
                    .unwrap(),
                None,
            )
            .unwrap();
            assert_eq!(pkt.hdr().len(), VSOCK_PKT_HDR_SIZE);
            // For TX packets, the host-owned buf is sized to the packet length (the bytes
            // copied out of guest memory at construction), not to the descriptor capacity.
            assert_eq!(pkt.buf().unwrap().len(), pkt.len() as usize);
        }

        // Test case: error on write-only hdr descriptor.
        {
            create_context!(test_ctx, handler_ctx);
            handler_ctx.guest_txvq.dtable[0]
                .flags
                .set(VRING_DESC_F_WRITE.try_into().unwrap());
            expect_asm_error!(tx, test_ctx, handler_ctx, VsockError::UnreadableDescriptor);
        }

        // Test case: header descriptor has insufficient space to hold the packet header.
        {
            create_context!(test_ctx, handler_ctx);
            handler_ctx.guest_txvq.dtable[0]
                .len
                .set(VSOCK_PKT_HDR_SIZE as u32 - 1);
            expect_asm_error!(tx, test_ctx, handler_ctx, VsockError::HdrDescTooSmall(_));
        }

        // Test case: zero-length TX packet.
        {
            create_context!(test_ctx, handler_ctx);
            set_pkt_len(0, &handler_ctx.guest_txvq.dtable[0], &test_ctx.mem);
            let mut pkt = VsockPacket::from_tx_virtq_head(
                &mut handler_ctx.handler.queues[1]
                    .iter(&test_ctx.mem)
                    .unwrap()
                    .next()
                    .unwrap(),
                None,
            )
            .unwrap();
            assert!(pkt.buf().is_none());
            assert!(pkt.buf_mut().is_none());
        }

        // Test case: TX packet has more data than we can handle.
        {
            create_context!(test_ctx, handler_ctx);
            set_pkt_len(
                MAX_PKT_BUF_SIZE as u32 + 1,
                &handler_ctx.guest_txvq.dtable[0],
                &test_ctx.mem,
            );
            expect_asm_error!(tx, test_ctx, handler_ctx, VsockError::InvalidPktLen(_));
        }

        // Test case: error on write-only buf descriptor.
        {
            create_context!(test_ctx, handler_ctx);
            handler_ctx.guest_txvq.dtable[1]
                .flags
                .set(VRING_DESC_F_WRITE.try_into().unwrap());
            expect_asm_error!(tx, test_ctx, handler_ctx, VsockError::UnreadableDescriptor);
        }

        // Test case: the buffer descriptor cannot fit all the data advertised by the
        // packet header `len` field.
        {
            create_context!(test_ctx, handler_ctx);
            set_pkt_len(8 * 1024, &handler_ctx.guest_txvq.dtable[0], &test_ctx.mem);
            handler_ctx.guest_txvq.dtable[1].len.set(4 * 1024);
            expect_asm_error!(tx, test_ctx, handler_ctx, VsockError::BufDescTooSmall);
        }
    }

    #[test]
    fn test_tx_packet_assembly_multi_desc() {
        const QSIZE: u16 = 4;
        let test_ctx = TestContext::new();
        let guest_txvq = GuestQ::new(GuestAddress(0x0060_0000), &test_ctx.mem, QSIZE);
        let mut queue = guest_txvq.create_queue();

        guest_txvq.dtable[0].set(
            0x0061_0000,
            VSOCK_PKT_HDR_SIZE as u32,
            virtio_bindings::virtio_ring::VRING_DESC_F_NEXT
                .try_into()
                .unwrap(),
            1,
        );
        guest_txvq.dtable[1].set(
            0x0061_1000,
            4 * 1024,
            virtio_bindings::virtio_ring::VRING_DESC_F_NEXT
                .try_into()
                .unwrap(),
            2,
        );
        guest_txvq.dtable[2].set(0x0061_2000, 4 * 1024, 0, 0);
        guest_txvq.avail.ring[0].set(0);
        guest_txvq.avail.idx.set(1);

        set_pkt_len(8 * 1024, &guest_txvq.dtable[0], &test_ctx.mem);

        let pkt = VsockPacket::from_tx_virtq_head(
            &mut queue.iter(&test_ctx.mem).unwrap().next().unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(pkt.len(), 8 * 1024);
        assert_eq!(pkt.buf().unwrap().len(), 8 * 1024);
    }

    #[test]
    fn test_rx_packet_assembly() {
        // Test case: successful RX packet assembly.
        {
            create_context!(test_ctx, handler_ctx);
            let pkt = VsockPacket::from_rx_virtq_head(
                &mut handler_ctx.handler.queues[0]
                    .iter(&test_ctx.mem)
                    .unwrap()
                    .next()
                    .unwrap(),
                None,
            )
            .unwrap();
            assert_eq!(pkt.hdr().len(), VSOCK_PKT_HDR_SIZE);
            assert_eq!(
                pkt.buf().unwrap().len(),
                handler_ctx.guest_rxvq.dtable[1].len.get() as usize
            );
        }

        // Test case: read-only RX packet header.
        {
            create_context!(test_ctx, handler_ctx);
            handler_ctx.guest_rxvq.dtable[0].flags.set(0);
            expect_asm_error!(rx, test_ctx, handler_ctx, VsockError::UnwritableDescriptor);
        }

        // Test case: RX descriptor head cannot fit the entire packet header.
        {
            create_context!(test_ctx, handler_ctx);
            handler_ctx.guest_rxvq.dtable[0]
                .len
                .set(VSOCK_PKT_HDR_SIZE as u32 - 1);
            expect_asm_error!(rx, test_ctx, handler_ctx, VsockError::HdrDescTooSmall(_));
        }
    }

    #[test]
    fn test_packet_hdr_accessors() {
        const SRC_CID: u64 = 1;
        const DST_CID: u64 = 2;
        const SRC_PORT: u32 = 3;
        const DST_PORT: u32 = 4;
        const LEN: u32 = 5;
        const TYPE: u16 = 6;
        const OP: u16 = 7;
        const FLAGS: u32 = 8;
        const BUF_ALLOC: u32 = 9;
        const FWD_CNT: u32 = 10;

        create_context!(test_ctx, handler_ctx);
        let mut pkt = VsockPacket::from_rx_virtq_head(
            &mut handler_ctx.handler.queues[0]
                .iter(&test_ctx.mem)
                .unwrap()
                .next()
                .unwrap(),
            None,
        )
        .unwrap();

        // Test field accessors.
        pkt.set_src_cid(SRC_CID)
            .set_dst_cid(DST_CID)
            .set_src_port(SRC_PORT)
            .set_dst_port(DST_PORT)
            .set_len(LEN)
            .set_type(TYPE)
            .set_op(OP)
            .set_flags(FLAGS)
            .set_buf_alloc(BUF_ALLOC)
            .set_fwd_cnt(FWD_CNT);

        assert_eq!(pkt.src_cid(), SRC_CID);
        assert_eq!(pkt.dst_cid(), DST_CID);
        assert_eq!(pkt.src_port(), SRC_PORT);
        assert_eq!(pkt.dst_port(), DST_PORT);
        assert_eq!(pkt.len(), LEN);
        assert_eq!(pkt.type_(), TYPE);
        assert_eq!(pkt.op(), OP);
        assert_eq!(pkt.flags(), FLAGS);
        assert_eq!(pkt.buf_alloc(), BUF_ALLOC);
        assert_eq!(pkt.fwd_cnt(), FWD_CNT);

        // Test individual flag setting.
        let flags = pkt.flags() | 0b1000;
        pkt.set_flag(0b1000);
        assert_eq!(pkt.flags(), flags);

        // Test packet header as-slice access.
        //

        assert_eq!(pkt.hdr().len(), VSOCK_PKT_HDR_SIZE);

        assert_eq!(
            SRC_CID,
            LittleEndian::read_u64(&pkt.hdr()[HDROFF_SRC_CID..])
        );
        assert_eq!(
            DST_CID,
            LittleEndian::read_u64(&pkt.hdr()[HDROFF_DST_CID..])
        );
        assert_eq!(
            SRC_PORT,
            LittleEndian::read_u32(&pkt.hdr()[HDROFF_SRC_PORT..])
        );
        assert_eq!(
            DST_PORT,
            LittleEndian::read_u32(&pkt.hdr()[HDROFF_DST_PORT..])
        );
        assert_eq!(LEN, LittleEndian::read_u32(&pkt.hdr()[HDROFF_LEN..]));
        assert_eq!(TYPE, LittleEndian::read_u16(&pkt.hdr()[HDROFF_TYPE..]));
        assert_eq!(OP, LittleEndian::read_u16(&pkt.hdr()[HDROFF_OP..]));
        assert_eq!(FLAGS, LittleEndian::read_u32(&pkt.hdr()[HDROFF_FLAGS..]));
        assert_eq!(
            BUF_ALLOC,
            LittleEndian::read_u32(&pkt.hdr()[HDROFF_BUF_ALLOC..])
        );
        assert_eq!(
            FWD_CNT,
            LittleEndian::read_u32(&pkt.hdr()[HDROFF_FWD_CNT..])
        );

        assert_eq!(pkt.hdr_mut().len(), VSOCK_PKT_HDR_SIZE);
        for b in pkt.hdr_mut() {
            *b = 0;
        }
        assert_eq!(pkt.src_cid(), 0);
        assert_eq!(pkt.dst_cid(), 0);
        assert_eq!(pkt.src_port(), 0);
        assert_eq!(pkt.dst_port(), 0);
        assert_eq!(pkt.len(), 0);
        assert_eq!(pkt.type_(), 0);
        assert_eq!(pkt.op(), 0);
        assert_eq!(pkt.flags(), 0);
        assert_eq!(pkt.buf_alloc(), 0);
        assert_eq!(pkt.fwd_cnt(), 0);
    }

    #[test]
    fn test_packet_buf() {
        create_context!(test_ctx, handler_ctx);
        let mut pkt = VsockPacket::from_rx_virtq_head(
            &mut handler_ctx.handler.queues[0]
                .iter(&test_ctx.mem)
                .unwrap()
                .next()
                .unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(
            pkt.buf().unwrap().len(),
            handler_ctx.guest_rxvq.dtable[1].len.get() as usize
        );
        assert_eq!(
            pkt.buf_mut().unwrap().len(),
            handler_ctx.guest_rxvq.dtable[1].len.get() as usize
        );

        for i in 0..pkt.buf().unwrap().len() {
            pkt.buf_mut().unwrap()[i] = (i % 0x100) as u8;
            assert_eq!(pkt.buf().unwrap()[i], (i % 0x100) as u8);
        }
    }

    #[test]
    fn test_tx_packet_data_is_owned_copy() {
        create_context!(test_ctx, handler_ctx);
        // Override the macro's default 1024-byte packet with a small payload we can read back.
        set_pkt_len(8, &handler_ctx.guest_txvq.dtable[0], &test_ctx.mem);

        let data_gpa = handler_ctx.guest_txvq.dtable[1].addr.get();
        let original = [0x11_u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        test_ctx
            .mem
            .write_slice(&original, GuestAddress(data_gpa))
            .unwrap();

        let pkt = VsockPacket::from_tx_virtq_head(
            &mut handler_ctx.handler.queues[1]
                .iter(&test_ctx.mem)
                .unwrap()
                .next()
                .unwrap(),
            None,
        )
        .unwrap();

        // Scribble over the guest source bytes after construction. The packet's view should
        // remain the snapshot taken at construction time, because the bytes were copied into
        // a host-owned buffer.
        test_ctx
            .mem
            .write_slice(&[0xff_u8; 8], GuestAddress(data_gpa))
            .unwrap();

        assert_eq!(pkt.buf().unwrap(), &original);
    }

    #[test]
    fn test_rx_packet_commit_buf_writes_to_guest() {
        create_context!(test_ctx, handler_ctx);
        let mut pkt = VsockPacket::from_rx_virtq_head(
            &mut handler_ctx.handler.queues[0]
                .iter(&test_ctx.mem)
                .unwrap()
                .next()
                .unwrap(),
            None,
        )
        .unwrap();

        // Backend writes payload into the bounce buffer and sets pkt.len().
        let payload = [0xab_u8, 0xcd, 0xef, 0x01, 0x23, 0x45];
        pkt.buf_mut().unwrap()[..payload.len()].copy_from_slice(&payload);
        pkt.set_len(payload.len() as u32);

        // Guest memory at the RX data GPA should not contain the payload yet.
        let data_gpa = handler_ctx.guest_rxvq.dtable[1].addr.get();
        let mut before = [0u8; 6];
        test_ctx
            .mem
            .read_slice(&mut before, GuestAddress(data_gpa))
            .unwrap();
        assert_eq!(&before, &[0u8; 6]);

        pkt.commit_buf(&test_ctx.mem).unwrap();

        // After commit_buf, the payload should be visible at the guest's data GPA.
        let mut after = [0u8; 6];
        test_ctx
            .mem
            .read_slice(&mut after, GuestAddress(data_gpa))
            .unwrap();
        assert_eq!(&after, &payload);
    }

    #[test]
    fn test_commit_buf_is_noop_for_zero_length_packet() {
        create_context!(test_ctx, handler_ctx);
        let pkt = VsockPacket::from_rx_virtq_head(
            &mut handler_ctx.handler.queues[0]
                .iter(&test_ctx.mem)
                .unwrap()
                .next()
                .unwrap(),
            None,
        )
        .unwrap();

        // The packet header was zeroed at construction, so pkt.len() == 0. commit_buf must
        // succeed without touching guest memory.
        assert_eq!(pkt.len(), 0);
        pkt.commit_buf(&test_ctx.mem).unwrap();
    }

    #[test]
    fn test_commit_buf_rejects_len_above_buf_capacity() {
        create_context!(test_ctx, handler_ctx);
        let mut pkt = VsockPacket::from_rx_virtq_head(
            &mut handler_ctx.handler.queues[0]
                .iter(&test_ctx.mem)
                .unwrap()
                .next()
                .unwrap(),
            None,
        )
        .unwrap();

        let cap = pkt.buf().unwrap().len() as u32;
        pkt.set_len(cap + 1);

        match pkt.commit_buf(&test_ctx.mem) {
            Err(VsockError::InvalidPktLen(n)) => assert_eq!(n, cap + 1),
            other => panic!("expected InvalidPktLen, got {other:?}"),
        }
    }
}
