import gradio as gr


def function():
    return


def interface():
    # define UI

    with gr.Blocks(css=".gradio-container {max-width: 1024px; margin: auto;}") as demo:
        # title
        gr.Markdown('# DesireQuest')
        gr.Markdown('[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) Text-to-3D Reference Project')
        gr.Markdown('[Our Code](https://github.com/arontaupe/stable-dreamfusion) DesireQuest')
        # inputs
        with gr.Tab("Default Options"):
            prompt = gr.Textbox(label="Prompt", max_lines=1, value="a DSLR photo of a koi fish")
            iters = gr.Slider(label="Iters", minimum=1000, maximum=20000, value=5000, step=100)
            seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)
        with gr.Tab("Advanced"):
            ws = gr.Textbox(label="Workspace \r\n (the folder where the results are saved)", max_lines=1,
                            value="a DSLR photo of a koi fish")
            lr = gr.Slider(label="Initial Learning Rate", minimum=1e-4, maximum=1e-2, value=1e-3, step=1e-4,
                           interactive=True)
            bb = gr.Dropdown(label="Backbone", choices=["grid", "vanilla"], value='grid', interactive=True)
            STEPS = gr.Slider(label="Steps \r\n (Nr. of Steps in an Epoch, we get one Vis per Epoch)", minimum=8,
                              maximum=32, value=8, step=2, interactive=True)
            guide = gr.Dropdown(label="Guidance", choices=["stable-diffusion", "clip"], value='stable-diffusion',
                                interactive=True)
            mesh = gr.Checkbox(label="Save Mesh", interactive=True, value=True)
        with gr.Tab("Ray settings"):
            cuda = gr.Checkbox(label="Use Cuda Raymarching", interactive=True, value=False)
            preset = gr.Checkboxgroup(label="Use Presets",
                                      choices=["0 : fp16, dir_text, cuda_ray", "02 : vanilla backbone"],
                                      interactive=True)

        with gr.Row():
            button = gr.Button('Train current Prompt')
            checkpoint = gr.Button('Pause and create Checkpoint', interactive=False)

        inputs = [prompt, iters, seed]

        # outputs
        with gr.Tab("Output"):
            with gr.Row():
                image = gr.Image(label="image", visible=True)
                depth_image = gr.Image(label="depth_image", visible=True)
                loss = gr.Plot(label="Loss Function", visible=True, interactive=False)
        with gr.Tab("Final Video"):
            video = gr.Video(label="video", visible=True)
            export_vid = gr.Button('Export Video')
        with gr.Tab("Final Mesh"):
            mesh = gr.Model3D(label="Final Mesh", visible=True)
            export_mesh = gr.Button('Export Mesh')
        logs = gr.Textbox(label="logging")

        outputs = [image, depth_image, video, mesh]

    # concurrency_count: only allow ONE running progress, else GPU will OOM.
    demo.queue(concurrency_count=1)
    # gr.Interface(fn=function(), inputs=inputs, outputs=outputs,
    #             title="DesireQuest", description="An App that lets you generate an object you desire in VR")

    demo.launch(show_error=True,
                show_tips=True,
                show_api=False,
                inbrowser=True,
                # server_port=7860,
                )


if __name__ == '__main__':
    interface()
