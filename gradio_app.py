import time

import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *

import gradio as gr
import gc
import matplotlib.pyplot as plt


print(f'[INFO] loading options..')

# fake config object, this should not be used in CMD, only allow change from gradio UI.
parser = argparse.ArgumentParser()
parser.add_argument('--text', default=None, help="text prompt")
# parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
# parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
parser.add_argument('--test', action='store_true', help="test mode")
parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
parser.add_argument('--workspace', type=str, default='trial_gradio')
parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=10000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='latest')
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=1024,
                    help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=64,
                    help="num steps sampled per ray (only valid when not using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=64,
                    help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16,
                    help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096,
                    help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
# model options
parser.add_argument('--bg_radius', type=float, default=1.4,
                    help="if positive, use a background model at sphere(bg_radius)")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
# network backbone
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, vanilla]")
# rendering resolution in training, decrease this if CUDA OOM.
parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")

### dataset options
parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
parser.add_argument('--dt_gamma', type=float, default=0,
                    help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
parser.add_argument('--dir_text', action='store_true',
                    help="direction-encode the text prompt, by appending front/side/back/overhead view")
parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
parser.add_argument('--angle_front', type=float, default=60,
                    help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")

### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=800, help="GUI width")
parser.add_argument('--H', type=int, default=800, help="GUI height")
parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
parser.add_argument('--light_theta', type=float, default=60,
                    help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

opt = parser.parse_args()

# default to use -O !!!
opt.fp16 = True
opt.dir_text = True
opt.cuda_ray = True
# opt.lambda_entropy = 1e-4
# opt.lambda_opacity = 0

if opt.backbone == 'vanilla':
    from nerf.network import NeRFNetwork
elif opt.backbone == 'grid':
    from nerf.network_grid import NeRFNetwork
else:
    raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] loading models..')

if opt.guidance == 'stable-diffusion':
    from nerf.sd import StableDiffusion

    guidance = StableDiffusion(device)
elif opt.guidance == 'clip':
    from nerf.clip import CLIP

    guidance = CLIP(device)
else:
    raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

print(f'[INFO] everything loaded!')

trainer = None
model = None

# define UI
with gr.Blocks(css=".gradio-container {max-width: 1024px; margin: auto;}") as demo:
    # title
    gr.Markdown('# DesireQuest')
    gr.Markdown('[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) Text-to-3D Reference Project')
    gr.Markdown('[Our Code](https://github.com/arontaupe/stable-dreamfusion) DesireQuest')
    # inputs
    with gr.Tab("Default Options"):
        prompt = gr.Textbox(label="Prompt", max_lines=1, value="a DSLR photo of a koi fish")
        iters = gr.Slider(label="Iters", minimum=100, maximum=20000, value=3000, step=100)
        seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)

    with gr.Tab("Advanced"):
        negative = gr.Textbox(label="Negative Prompt", max_lines=1, value="")
        suppress_face = gr.Checkbox(label='Suppress Face', interactive=True)
        ws = gr.Textbox(label="Workspace \r\n (the folder where the results are saved)", max_lines=1,
                        value="gradio_test", interactive=True)
        lr = gr.Slider(label="Initial Learning Rate", minimum=1e-4, maximum=1e-2, value=1e-3, step=1e-4,
                       interactive=True)
        checkpoint = gr.Dropdown(label="Use Existing Checkpoints", choices=["latest", "scratch"], value='latest',
                                 interactive=True)
        steps_per_epoch = gr.Slider(label="Steps \r\n (Nr. of Steps in an Epoch, we get one Vis per Epoch)", minimum=8,
                                    maximum=32, value=8, step=2, interactive=True)
        guide = gr.Dropdown(label="Guidance", choices=["stable-diffusion", "clip"], value='stable-diffusion',
                            interactive=True)
        mesh = gr.Checkbox(label="Save Mesh", interactive=True, value=True)
        albedo = gr.Checkbox(label="Use Albedo as Color (uglier and faster)", interactive=True, value=False)
        jitter = gr.Checkbox(label="Add jitter to Camera Poses", interactive=True, value=False)

    with gr.Tab("Ray settings"):
        cuda = gr.Checkbox(label="Use Cuda Raymarching", interactive=True, value=False)
        bb_preset = gr.Dropdown(label="Use Presets",
                                  choices=['none', "grid", "vanilla"],
                                value='none',
                                  interactive=True)
    with gr.Tab("Optimizer"):
        gr.Markdown('The paper uses standard ADAM. We DO NOT fuck with that.')

    flags = gr.Textbox(label="The default Values", value=opt, visible=True, interactive=False)

    with gr.Row():
        button = gr.Button('Train current Prompt')
        # TODO make this button actually do something. i would like best if it takes checkpoints regularly and pauses gracefully
        checkpoint_button = gr.Button('Pause and create Checkpoint', interactive=False)
    # define here to give at button press
    inputs = [prompt, iters, seed, negative, suppress_face, checkpoint, lr, bb_preset, mesh, ws, albedo, guide, jitter]

    # outputs
    with gr.Tab("Output"):
        with gr.Row():
            image = gr.Image(label="image", visible=True, interactive=False)
            depth_image = gr.Image(label="depth_image", visible=True, interactive=False)
            # TODO not deemed important, i think seeing loss isnt much use (and i cant get it to output loss)
            #loss = gr.Plot(label="Loss Function", visible=True, interactive=False)
    with gr.Tab("Final Video"):
        video = gr.Video(label="video", visible=True, interactive=False)
        # TODO make button work
        export_vid = gr.Button('Export Video', interactive=False)
    with gr.Tab("Final Mesh"):
        # TODO somehow bind the output mehs to gradio so it can be displayed
        mesh_viz = gr.Model3D(label="Final Mesh", visible=True, interactive=False)
        export_mesh = gr.Button('Export Mesh', interactive=False)


    with gr.Row():
        memory = gr.Textbox(label="memory watcher")
        current_lr = gr.Number(label="Current Learning Rate", interactive=False, visible=True)
        time_elapsed = gr.Number(label="Time Elapsed", interactive=False, visible=True)
        time_eta = gr.Number(label="Current ETA estimate", interactive=False, visible=True)
        time_last_epoch = gr.Number(label="Time last Epoch", interactive=False, visible=True)
        avg_epoch_time = gr.Number(label="Average time per epoch", interactive=False, visible=True)

    logs = gr.Textbox(label="logging")

    #define outputs as list to give at buttonpress
    outputs = [image, depth_image, video, current_lr, logs, memory, flags, time_elapsed, time_last_epoch, mesh_viz, avg_epoch_time, time_eta]


    def submit(text, iters, seed, negative, suppress_face, checkpoint, lr, bb_preset, mesh, ws, albedo, guide, jitter):

        global trainer, model

        # seed
        opt.seed = seed
        opt.text = text
        opt.iters = iters
        opt.negative = negative
        opt.suppress_face = suppress_face
        opt.uniform_sphere_rate = 0.5
        opt.lambda_smooth = 0
        opt.ckpt = checkpoint
        opt.workspace = ws
        opt.lr = lr
        opt.jitter_pose = jitter
        opt.albedo = albedo
        opt.guidance = guide
        if opt.albedo:
            opt.albedo_iters = opt.iters

        if bb_preset == 'grid':
            opt.O = True
            opt.fp16 = True
            opt.dir_text = True
            opt.cuda_ray = True

        if bb_preset == 'vanilla':
            opt.O2 = True
            # only use fp16 if not evaluating normals (else lead to NaNs in training...)
            if opt.albedo:
                opt.fp16 = True
            opt.dir_text = True
            opt.backbone = 'vanilla'

        if mesh:
            opt.save_mesh = True

        print(opt)
        yield{
            flags: gr.update(label='Values used this run', value=opt)
        }

        seed_everything(seed)

        # clean up
        if trainer is not None:
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print('[INFO] clean up!')

        # simply reload everything...
        model = NeRFNetwork(opt)
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                  lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer,
                          ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt,
                          eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        # train (every ep only contain 8 steps, so we can get some vis every ~10s)
        STEPS = steps_per_epoch.value
        max_epochs = np.ceil(opt.iters / STEPS).astype(np.int32)

        # we have to get the explicit training loop out here to yield progressive results...
        loader = iter(valid_loader)

        start_t = time.time()
        avg_time_per_epoch = 0

        for epoch in range(max_epochs):
            epoch_start_t = time.time()
            trainer.train_gui(train_loader, step=STEPS)

            # manual test and get intermediate results
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(valid_loader)
                data = next(loader)

            trainer.model.eval()

            if trainer.ema is not None:
                trainer.ema.store()
                trainer.ema.copy_to()

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=trainer.fp16):
                    preds, preds_depth = trainer.test_step(data, perturb=False)

            if trainer.ema is not None:
                trainer.ema.restore()

            pred = preds[0].detach().cpu().numpy()
            pred_depth = preds_depth[0].detach().cpu().numpy()

            pred = (pred * 255).astype(np.uint8)

            free, total = torch.cuda.mem_get_info(device=0)
            total = total / 1024 ** 3
            free = free / 1024 ** 3
            used = total - free
            mem_text = f'USED {round(used, 2)} \r\n' \
                       f' FREE {round(free, 2)} \r\n' \
                       f' TOTAL {round(total, 2)}\r\n '

            avg_time_per_epoch += (time.time() - start_t) /(epoch +1)
            yield {
                image: gr.update(value=pred, visible=True),
                depth_image: gr.update(value=pred_depth, visible=True),
                video: gr.update(visible=False),
                current_lr: gr.update(value=round(trainer.optimizer.param_groups[0]['lr'], 5), visible=True),
                logs: f"training iters: {epoch * STEPS} / {iters}, lr: {trainer.optimizer.param_groups[0]['lr']:.6f}",
                memory: mem_text,
                time_elapsed: gr.update(value=round((time.time() - start_t)/60,2)),
                time_last_epoch: gr.update(value=round((time.time() - epoch_start_t)/60,2)),
                time_eta: gr.update(value=round(iters -(epoch * STEPS) * (time.time() - epoch_start_t)/60,2)),
                avg_epoch_time: gr.update(value=round(avg_time_per_epoch/60, 2)),

                # TODO Make avg time happening
            }

        # test
        trainer.test(test_loader)

        results = glob.glob(os.path.join(opt.workspace, 'results', '*rgb*.mp4'))
        mesh_obj = glob.glob(os.path.join(opt.workspace, 'results', '*.obj'))

        # TODO figure out how to save the mesh once it is generated
        assert results is not None, "cannot retrieve results!"
        results.sort(key=lambda x: os.path.getmtime(x))  # sort by mtime

        end_t = time.time()

        yield {
            image: gr.update(visible=True),
            depth_image: gr.update(visible=True),
            video: gr.update(value=results[-1], visible=True),
            mesh_viz: gr.update(value=mesh_obj, visible=True),
            logs: f"Generation Finished in {(end_t - start_t) / 60:.4f} minutes!",
        }


    button.click(
        fn=submit,
        inputs=inputs,
        outputs=outputs)

# concurrency_count: only allow ONE running progress, else GPU will OOM.
demo.queue(concurrency_count=1)

# IMPORTANT: if the program runs, it reacts sensitive to shutdowns. let it gracefully shutdown instead of killing it.
# saves you from getting OOM
demo.launch(show_error=True,
            show_tips=False,
            show_api=False,
            #inbrowser=True,
            # server_port=7860,
            #share=True,
            )
