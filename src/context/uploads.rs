//! What [`Context::write_buffer`](super::Context::write_buffer) crosses to the GPU
//! through, kept from frame to frame.
//!
//! On every backend wgpu implements itself, `Queue::write_buffer` makes a buffer
//! per call and frees it once the GPU has copied from it, and a frame makes dozens
//! of calls. Here the bytes go into a [`StagingBelt`]'s chunks, which are used again
//! as the GPU finishes with them, and the copies out of them wait in an encoder
//! that goes first in the next submission. That is where `Queue::write_buffer`'s
//! land too, so a write still reaches everything in that submission, and of two
//! writes to one place the later still wins. Only `Context::submit` knows to send
//! them, so a submission made on the queue directly leaves them for the next.

use std::iter;
use wgpu::util::StagingBelt;

/// What a chunk holds, and the most one write takes through the belt. A larger
/// write is rare (a mesh, a texture's worth of data) and a chunk is never freed,
/// so it goes through the queue and takes its memory back afterwards.
const CHUNK_SIZE: wgpu::BufferAddress = 256 * 1024;

/// The most a submission takes through the belt, which bounds what the belt keeps
/// at this much for each submission in flight.
const SUBMISSION_BUDGET: wgpu::BufferAddress = 4 * 1024 * 1024;

pub(super) struct Uploads {
    device: wgpu::Device,
    /// `None` on the browser's WebGPU: there `Queue::write_buffer` is one call into
    /// JavaScript and makes no buffer, and a mapping would cross three more times.
    belt: Option<StagingBelt>,
    encoder: Option<wgpu::CommandEncoder>,
    /// What the encoder's copies carry.
    carrying: wgpu::BufferAddress,
}

impl Uploads {
    pub(super) fn new(device: &wgpu::Device, backend: wgpu::Backend) -> Self {
        Self {
            device: device.clone(),
            belt: (backend != wgpu::Backend::BrowserWebGpu)
                .then(|| StagingBelt::new(device.clone(), CHUNK_SIZE)),
            encoder: None,
            carrying: 0,
        }
    }

    pub(super) fn write(
        &mut self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[u8],
    ) {
        let bytes = data.len() as wgpu::BufferAddress;
        let fits = bytes <= CHUNK_SIZE && self.carrying + bytes <= SUBMISSION_BUDGET;
        let (Some(belt), Some(size), true) =
            (self.belt.as_mut(), wgpu::BufferSize::new(bytes), fits)
        else {
            // The queue's writes land before every command buffer of a submission,
            // the encoder's copies with them: those go first to stay first.
            if let Some(copies) = self.take_copies() {
                queue.submit(iter::once(copies));
            }
            queue.write_buffer(buffer, offset, data);
            return;
        };
        let device = &self.device;
        let encoder = self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("kiss3d_uploads"),
            })
        });
        belt.write_buffer(encoder, buffer, offset, size)
            .copy_from_slice(data);
        self.carrying += bytes;
    }

    /// The copies waiting, which have to go first in the next submission.
    pub(super) fn take_copies(&mut self) -> Option<wgpu::CommandBuffer> {
        let encoder = self.encoder.take()?;
        if let Some(belt) = &mut self.belt {
            belt.finish_and_recall_on_submit(&encoder);
        }
        self.carrying = 0;
        Some(encoder.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Context;
    use crate::test_gpu::on_gpu;

    const SIZE: usize = 2 * CHUNK_SIZE as usize;

    fn target(ctxt: &Context) -> wgpu::Buffer {
        ctxt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uploads_test_target"),
            size: SIZE as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// The first bytes of `buffer`, as a submission made after the writes sees them.
    fn first_word(ctxt: &Context, buffer: &wgpu::Buffer) -> u32 {
        let read = ctxt.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uploads_test_read"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctxt.create_command_encoder(None);
        encoder.copy_buffer_to_buffer(buffer, 0, &read, 0, 4);
        ctxt.submit(iter::once(encoder.finish()));
        read_word(ctxt, &read)
    }

    fn read_word(ctxt: &Context, read: &wgpu::Buffer) -> u32 {
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        ctxt.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device is alive");
        let view = read
            .slice(..)
            .get_mapped_range()
            .expect("mapped by the poll above");
        let word = u32::from_le_bytes([view[0], view[1], view[2], view[3]]);
        drop(view);
        read.unmap();
        word
    }

    #[test]
    fn a_write_reaches_a_command_recorded_before_it() {
        on_gpu(4, 4, async |_surface| {
            let ctxt = Context::get();
            let buffer = target(&ctxt);
            let read = ctxt.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uploads_test_read"),
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = ctxt.create_command_encoder(None);
            encoder.copy_buffer_to_buffer(&buffer, 0, &read, 0, 4);
            ctxt.write_buffer(&buffer, 0, &7u32.to_le_bytes());
            ctxt.submit(iter::once(encoder.finish()));
            assert_eq!(read_word(&ctxt, &read), 7);
        });
    }

    #[test]
    fn of_two_writes_the_later_wins_whichever_way_each_went() {
        on_gpu(4, 4, async |_surface| {
            let ctxt = Context::get();
            let buffer = target(&ctxt);
            let large = |word: u32| -> Vec<u8> {
                word.to_le_bytes()
                    .iter()
                    .copied()
                    .cycle()
                    .take(SIZE)
                    .collect()
            };

            ctxt.write_buffer(&buffer, 0, &1u32.to_le_bytes());
            ctxt.write_buffer(&buffer, 0, &2u32.to_le_bytes());
            assert_eq!(first_word(&ctxt, &buffer), 2, "belt then belt");

            ctxt.write_buffer(&buffer, 0, &3u32.to_le_bytes());
            ctxt.write_buffer(&buffer, 0, &large(4));
            assert_eq!(first_word(&ctxt, &buffer), 4, "belt then queue");

            ctxt.write_buffer(&buffer, 0, &large(5));
            ctxt.write_buffer(&buffer, 0, &6u32.to_le_bytes());
            assert_eq!(first_word(&ctxt, &buffer), 6, "queue then belt");
        });
    }

    #[test]
    fn past_the_budget_writes_still_arrive_in_order() {
        on_gpu(4, 4, async |_surface| {
            let ctxt = Context::get();
            let buffer = target(&ctxt);
            let chunk = vec![0u8; CHUNK_SIZE as usize];
            for _ in 0..=SUBMISSION_BUDGET / CHUNK_SIZE {
                ctxt.write_buffer(&buffer, 0, &chunk);
            }
            ctxt.write_buffer(&buffer, 0, &9u32.to_le_bytes());
            assert_eq!(first_word(&ctxt, &buffer), 9);
        });
    }

    #[test]
    fn frames_of_small_writes_make_no_new_buffer_once_warm() {
        on_gpu(4, 4, async |_surface| {
            let ctxt = Context::get();
            let buffer = target(&ctxt);
            let frame = |word: u32| {
                for _ in 0..32 {
                    ctxt.write_buffer(&buffer, 0, &word.to_le_bytes());
                }
                assert_eq!(first_word(&ctxt, &buffer), word);
            };
            let buffers = || {
                ctxt.instance
                    .generate_report()
                    .expect("not the browser's WebGPU")
                    .hub
                    .buffers
                    .num_allocated
            };
            for word in 0..4 {
                frame(word);
            }
            let warm = buffers();
            for word in 4..64 {
                frame(word);
            }
            // Each `frame` makes one read buffer, dropped as it ends.
            assert!(buffers() <= warm + 1, "{} after {warm}", buffers());
        });
    }

    #[test]
    fn nothing_written_means_no_copies() {
        on_gpu(4, 4, async |_surface| {
            let ctxt = Context::get();
            ctxt.submit(iter::empty());
            let buffer = target(&ctxt);
            assert!(ctxt.uploads().take_copies().is_none());
            ctxt.write_buffer(&buffer, 0, &[]);
            assert!(
                ctxt.uploads().take_copies().is_none(),
                "an empty write crosses nothing"
            );
        });
    }
}
