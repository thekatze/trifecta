#[forbid(unsafe_code)]

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(run());
}

// right handed coordinate system
// +x -> right
// +y -> up
// +z -> out of the screen towards my face (hither)

fn cube_vertices() -> ([MeshVertex; 8], [u32; 36]) {
    let vertices = [
        MeshVertex {
            position: glam::Vec3::new(-0.5, 0.5, 0.5),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
        },
        MeshVertex {
            position: glam::Vec3::new(0.5, 0.5, 0.5),
            color: glam::Vec3::new(0.0, 1.0, 0.0),
        },
        MeshVertex {
            position: glam::Vec3::new(0.5, 0.5, -0.5),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
        },
        MeshVertex {
            position: glam::Vec3::new(-0.5, 0.5, -0.5),
            color: glam::Vec3::new(1.0, 0.0, 1.0),
        },
        MeshVertex {
            position: glam::Vec3::new(-0.5, -0.5, 0.5),
            color: glam::Vec3::new(1.0, 0.0, 0.0),
        },
        MeshVertex {
            position: glam::Vec3::new(0.5, -0.5, 0.5),
            color: glam::Vec3::new(0.0, 1.0, 0.0),
        },
        MeshVertex {
            position: glam::Vec3::new(0.5, -0.5, -0.5),
            color: glam::Vec3::new(0.0, 0.0, 1.0),
        },
        MeshVertex {
            position: glam::Vec3::new(-0.5, -0.5, -0.5),
            color: glam::Vec3::new(1.0, 0.0, 1.0),
        },
    ];

    let indices: [u32; 36] = [
        0, 1, 2, // top
        2, 3, 0, // top
        0, 4, 1, // front
        1, 4, 5, // front
        1, 5, 2, // right
        2, 5, 6, // right
        2, 6, 3, // back
        3, 6, 7, // back
        3, 7, 0, // left
        0, 7, 4, // left
        7, 6, 4, // bottom
        4, 6, 5, // bottom
    ];

    (vertices, indices)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    position: glam::Vec3,
    color: glam::Vec3,
}

impl MeshVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

struct Camera {
    position: glam::Vec3,
    target: glam::Vec3,
    up: glam::Vec3,
    aspect: f32,
    fov: f32,
    clip: std::ops::Range<f32>,
}

impl Camera {
    fn build_projection_matrix(&self) -> glam::Mat4 {
        // let view = glam::Mat4::look_to_rh(self.position, self.direction, self.up);
        let view = glam::Mat4::look_at_rh(self.position, self.target, self.up);

        let projection = glam::Mat4::perspective_rh(
            (self.fov / 360.0) * std::f32::consts::TAU,
            self.aspect,
            self.clip.start,
            self.clip.end,
        );

        projection * view
    }
}

use glam::u32;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            console_error_panic_hook::set_once();

            tracing_log::LogTracer::init().expect("log redirect to tracing failed");
            tracing_wasm::set_as_global_default();
        } else {
            tracing_subscriber::fmt::init();
        }
    }

    const WINDOW_WIDTH: u32 = 800;
    const WINDOW_HEIGHT: u32 = 450;

    let (window, event_loop) = {
        let event_loop = winit::event_loop::EventLoop::new().expect("creating event loop failed");
        let window = winit::window::WindowBuilder::new()
            .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
            }))
            .build(&event_loop)
            .expect("building window failed");

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|js_window| js_window.document()?.body())
                .and_then(|body| {
                    let canvas = web_sys::Element::from(window.canvas()?);
                    canvas
                        .set_attribute("width", WINDOW_WIDTH.to_string().as_str())
                        .ok()?;
                    canvas
                        .set_attribute("height", WINDOW_HEIGHT.to_string().as_str())
                        .ok()?;

                    // make dev more bearable
                    body.style()
                        .set_property("background-color", "#2e2e2e")
                        .ok()?;

                    body.append_child(&canvas).ok()
                })
                .expect("failed to add canvas to body");
        }

        (window, event_loop)
    };

    let (surface, device, queue, mut configuration) = async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let surface = instance
            .create_surface(&window)
            .expect("surface couldn't be created, check if webgpu is supported");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("adapter request failed");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("render_device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("device request failed");

        let capabilities = surface.get_capabilities(&adapter);

        let format = if capabilities
            .formats
            .contains(&wgpu::TextureFormat::Bgra8UnormSrgb)
        {
            wgpu::TextureFormat::Bgra8UnormSrgb
        } else {
            wgpu::TextureFormat::Bgra8Unorm
        };

        let configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            present_mode: wgpu::PresentMode::AutoVsync,
            format,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: WINDOW_WIDTH,
            height: WINDOW_HEIGHT,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        tracing::info!("configuring surface with {:?}", configuration);

        surface.configure(&device, &configuration);
        (surface, device, queue, configuration)
    }
    .await;

    let (cube_vertices, cube_indices) = cube_vertices();

    let (vertex_buffer, index_buffer) = {
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex_buffer"),
            size: (std::mem::size_of::<MeshVertex>() * cube_vertices.len()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index_buffer"),
            size: (std::mem::size_of::<u32>() * cube_indices.len()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&vertex_buffer, 0, bytemuck::bytes_of(&cube_vertices));
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&cube_indices));

        (vertex_buffer, index_buffer)
    };

    let mut camera = Camera {
        position: [1.0, 1.0, 2.0].into(),
        target: glam::Vec3::ZERO,
        up: glam::Vec3::Y,
        aspect: configuration.width as f32 / configuration.height as f32,
        fov: 95.0,
        clip: (0.1..100.0),
    };

    let (camera_buffer, camera_bind_group_layout, camera_bind_group) = {
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buffer"),
            size: std::mem::size_of::<glam::Mat4>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        (camera_buffer, camera_bind_group_layout, camera_bind_group)
    };

    let render_pipeline = {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[MeshVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: configuration.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    };

    let mut time = 0.0;

    {
        use winit::event::{Event, WindowEvent};

        event_loop
            .run(|event, context| match event {
                Event::WindowEvent { event, window_id } => {
                    if window_id == window.id() {
                        match event {
                            WindowEvent::CloseRequested => context.exit(),
                            WindowEvent::Resized(new_size) => {
                                camera.aspect = new_size.width as f32 / new_size.height as f32;

                                cfg_if::cfg_if! {
                                    if #[cfg(target_arch = "wasm32")] {
                                        configuration.width = new_size.width / 2;
                                        configuration.height = new_size.height / 2;
                                    } else {
                                        configuration.width = new_size.width;
                                        configuration.height = new_size.height;
                                    }
                                }

                                surface.configure(&device, &configuration);
                            },
                            WindowEvent::RedrawRequested => {
                                time += 0.01;

                                camera.position = glam::Vec3::from([f32::sin(time), 1.0, f32::cos(time)]);
                                queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera.build_projection_matrix()));

                                let mut encoder = device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: Some("command_encoder"),
                                    },
                                );

                                let output = match surface.get_current_texture() {
                                    Ok(output) => output,
                                    Err(err) => match err {
                                        wgpu::SurfaceError::Timeout
                                        | wgpu::SurfaceError::Outdated => {
                                            tracing::warn!(?err);
                                            // just try again next frame
                                            return;
                                        }
                                        wgpu::SurfaceError::Lost => {
                                            tracing::warn!(?err, "surface lost, skipping frame");
                                            // reconfigure and just try again next frame
                                            surface.configure(&device, &configuration);
                                            return;
                                        }
                                        wgpu::SurfaceError::OutOfMemory => {
                                            tracing::error!(?err, "gpu out of memory: couldn't allocate next frame buffer");
                                            context.exit();
                                            return;
                                        }
                                    },
                                };

                                let view =
                                    output.texture.create_view(&wgpu::TextureViewDescriptor {
                                        array_layer_count: None,
                                        label: Some("render_target"),
                                        format: Some(configuration.format),
                                        aspect: wgpu::TextureAspect::All,
                                        dimension: Some(wgpu::TextureViewDimension::D2),
                                        base_mip_level: 0,
                                        mip_level_count: None,
                                        base_array_layer: 0,
                                    });

                                {
                                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("my first render pass"),
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.6, g: 0.7, b: 0.9, a: 0.0 }),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                        depth_stencil_attachment: None,
                                    });

                                    render_pass.set_pipeline(&render_pipeline);
                                    render_pass.set_bind_group(0, &camera_bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                                    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                                    render_pass.draw_indexed(0..cube_indices.len() as u32, 0, 0..1);
                                }

                                queue.submit(Some(encoder.finish()));
                                output.present();
                            }

                            _ => {}
                        }
                    }
                }
                Event::AboutToWait => {
                    // todo: add frame limiting here
                    window.request_redraw();
                }
                _ => {}
            })
            .expect("event loop failed");
    }
}
