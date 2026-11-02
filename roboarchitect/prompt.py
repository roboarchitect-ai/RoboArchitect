from __future__ import annotations

import sys
from typing import Mapping

from roboarchitect.harness.motion import make_controller
from roboarchitect.inputs import CODE_ROOT, Setup

from auxiliary.common import MODELS


REVISION_PROMPT = (
    "You are a robotics engineer. Your job is to improve one Python program, `harness.py`, which is the robot's control agent: every step of an episode it receives what the robot can see and returns the numbers that move the robot. An episode is one attempt at one task, from reset until the robot succeeds or runs out of steps.\n"
    '\n'
    'Several frozen models are available to `harness.py`. One is a robot policy, trained to turn a camera picture plus a task sentence into robot motion; the others see or read. You cannot change any of them and you must not train or fine-tune anything. How you use them, and whether you use them at all, is yours to decide: the numbers you return can come from a model, from arithmetic you wrote, or from any mixture.\n'
    '\n'
    'Everything between the observation arriving and the action leaving is yours to change.\n'
    '\n'
    '\n'
    '## What to aim at\n'
    'Make the robot succeed on more episodes. Work out what is going wrong first, from the recordings described below, and only then change the code. Say what the failure is before you fix it.\n'
    '\n'
    'Whatever you change must be a general improvement to the robot agent. It has to keep working on tasks and scenes you have never seen, not just the ones in these recordings.\n'
    '\n'
    'How much you change is your call -- one line or a rewrite. Write a short note of what you changed and why into `../workspace/CHANGES.md`: what you saw in the recordings, what you concluded, and what you did about it. Finish by saying the same thing in your last message.\n'
    '\n'
    '## The one file that runs, and where everything else goes\n'
    '`harness.py` in your current directory is the whole robot agent, and it is the only file that runs. Edit it in place. Keep the agent in that one file only.\n'
    '\n'
    'Keep `create_harness()`, the `HARNESS_CONFIG` dict and the `act(request, tools)` signature. `HARNESS_CONFIG` keeps its two lists, and no name may be in both. A name in `required_capabilities` is loaded every time. A name in `optional_capabilities` is loaded only when your file mentions it. A name in neither is never loaded and calling it fails. `"vla"` is the robot policy and follows the same rule, so move it to `optional_capabilities` if you stop calling it.\n'
    '\n'
    'Everything else you produce -- notes, analysis scripts, plots, extracted data, rendered pictures -- goes in `../workspace/`. That folder is yours, put scratch files there rather than in `/tmp`.\n'
    '\n'
    '<<ACROSS_EPISODES>>\n'
    '\n'
    '\n'
    '## The interface (read `harness.py` first)\n'
    "`act(request, tools)` is called once per step and must return exactly one `ActionChunk`. Its `spec` must be this setup's own action spec. Its `start_step` must equal `request.observation.step_index`. Its values must obey this setup's limits, which the last section lists. The class is defined in `roboarchitect/protocol/action.py`.\n"
    '\n'
    '`request.observation` is everything the robot can see this step:\n'
    '  - `.instruction` -- the task sentence, word for word as the task gives it.\n'
    '  - `.cameras` -- a mapping from camera name to a camera picture.\n'
    "  - `.proprioception` -- the robot's own readings, typically its gripper pose and how open the gripper is. Which entries exist depends on the setup. Look one up by name with `.by_name(...)`, or read `.vectors`, where each entry has a `.spec` describing what its numbers mean and `.values` holding them.\n"
    '  - `.step_index` -- which step of the episode this is.\n'
    'There is nothing else in it. No object positions, no goal, no score.\n'
    '\n'
    "Each camera has `.rgb` (uint8, height x width x 3). Where this setup has depth sensing it also has `.depth_m` (float32, height x width, in metres, one value per colour pixel), `.depth_valid`, `.intrinsics` (3x3) and `.camera_to_world` (4x4). Those four together let you turn a pixel into a point in the robot's own coordinates.\n"
    '\n'
    'The exact definitions are in `roboarchitect/harness/api.py` (the request and result types), `roboarchitect/harness/tools.py` (the `tools` object) and `roboarchitect/protocol/observation.py` (the observation). Read them rather than guessing.\n'
    '\n'
    '\n'
    '## The models you can call\n'
    'Every call goes through the `tools` object. `roboarchitect/harness/tools.py` is the implementation and `roboarchitect/harness/api.py` holds every request and result type, with the exact fields and the checks each one applies. Read both before writing a call.\n'
    '\n'
    '- `tools.vla(...)` -- the robot policy. You can build the whole agent around it, use it on some steps only, or never call it at all. Three of its arguments are worth knowing about before you read the file:\n'
    '  - `instruction=` is the sentence the policy is given. The policy service feeds it to the model only on a step where it actually runs the model, so changing the text on any other step does nothing until then. See `refresh` next, and read `policies/` for the service this setup uses.\n'
    "  - `refresh=True` throws away whatever the policy service had left over and makes it run the model again on this step's picture. The service runs the model once, gets several future actions back, and hands them out one per step; that is why a new instruction can sit unused for several steps. A refresh is a change in behaviour, not just a recomputation: anything the discarded actions were carrying is gone with them.\n"
    "  - `context=` is a tuple of strings sent to the policy service. It is ignored unless the last section says this setup's policy reads it.\n"
    '  - If you want the policy to know something, it has to be in the instruction text or in the observation.\n'
    '\n'
    '- `tools.vision(...)` -- a text model that can see pictures. Describe a scene, compare two views, judge whether something is done.\n'
    '\n'
    '- `tools.language(...)` -- a text model. It cannot see pictures.\n'
    '\n'
    '- `tools.detect(...)` -- an object detector. Give it a picture and words; it returns boxes with labels and scores.\n'
    '\n'
    '- `tools.point(...)` -- a pointer. Give it a picture and an instruction; it returns pixel coordinates.\n'
    '\n'
    '- `tools.segment(...)` -- a segmenter. Give it a picture and at least one box or point; it returns outlines.\n'
    '\n'
    'Check with `tools.has("detection")` and so on before calling; the last section lists which models are available here. Calling one that is not running raises `ToolUnavailableError`, and so does a call that fails for any other reason, such as a timeout or a malformed request. Any exception that escapes `act` ends that episode and it counts as a failure, so guard your calls.\n'
    '\n'
    "`tools.record(event_type, status, detail, capability, result)` writes a line into this episode's recording. Every model call is recorded for you, answer included, so you do not have to log those yourself -- use `record` for your own decisions: what you concluded, what you chose to do, and why.\n"
    '\n'
    '\n'
    '## Two ready-made modules you can import\n'
    'Your harness runs with this project on its import path, so `from roboarchitect...` works. numpy and PIL are available; no simulator package and no PyYAML are, so `benchmarks` and `policies` will not import. Two modules are there for you to use, and both are plain arithmetic with no hidden state.\n'
    '\n'
    "- `roboarchitect.harness.geometry` -- turns a pixel into a point in the robot's own coordinates, and back. Call `has_3d(camera)` first: it is False where this setup has no depth. Then `depth_at`, `pixel_to_world`, `world_to_pixel`, `point_cloud`. `pixel_to_world` returns `None` rather than raising when there is no usable depth at that pixel, so check the result. The module already accounts for which way up this setup stores its picture.\n"
    '\n'
    "- `roboarchitect.harness.motion` -- ready-made movement commands you can return instead of a model's answer. `make_controller(spec)` gives you `move_to`, `nudge`, `hold` and `set_gripper`; each returns a float32 array of action numbers for one step, which you wrap in an `ActionChunk` yourself. `make_controller` returns `None` where the action layout is not one it can drive, so check for that; the last section says whether it can drive this setup. On a setup whose action is a change from the current pose these commands leave the wrist rotation alone; on a setup whose action is a target pose they reuse the rotation from the last chunk you passed to `controller.note(chunk)`.\n"
    '\n'
    "These commands are a convenience, not a limit. You may also compute the action numbers yourself and return them, as long as the chunk carries this setup's own spec and its values obey this setup's limits.\n"
    'Read both modules before using them.\n'
    '\n'
    '\n'
    '## If you call the policy, call it on every step\n'
    'The policy service remembers which step it last produced an action for and refuses a request that jumps ahead, with `policy_act: previous action is not observed as executed`. Do not try to remove that check.\n'
    '\n'
    'So there are exactly two shapes that work: call `tools.vla` on every step, and on any step return something else instead of its action if you want to -- or never call it at all for the whole episode. Mixing the two, by calling it, skipping some steps and calling it again, fails every episode. Throwing away an action you asked for costs only the time it took to compute.\n'
    '\n'
    'None of this applies to the other models. You can call or skip those freely, on any step.\n'
    '\n'
    '\n'
    '## The recordings are your evidence\n'
    '`../feedback.json` gives you the folders that hold them.\n'
    '  - `current_harness_recordings` -- the recordings of the `harness.py` you have now, running the whole episode set.\n'
    '  - `rejected_revision_recordings` -- present only sometimes. It is a change that was already tried and did not do better, with its own full recordings in the same shape.\n'
    '\n'
    'Read them yourself with Bash, Grep and Read, and analyse them with python. Each episode folder holds:\n'
    "  - `trace.jsonl` : line 0 names the task, says whether the episode succeeded and how it ended, and lists every camera with its size and the picture file that holds it. Every later line is one step, with `step`, `instruction` (what the policy was given), `action` (the numbers that were returned), `state` (the robot's own readings that step: gripper pose, and where the setup reports it, how far apart the fingers are), `depth` (per camera: how much of the depth picture was usable, and its smallest, middle and largest distance) and `events`.\n"
    "  - Each event is one thing that happened that step. A model call carries `result`, which holds what that model actually answered -- the detector's boxes with labels and scores, the pointer's pixels, a summary of each mask, the text a text model wrote. Events your harness wrote with `tools.record(...)` appear the same way.\n"
    '  - `camera-<name>.mp4` : every rendered frame of that camera for the whole episode. Pull any step out with ffmpeg, for example `ffmpeg -i camera-main.mp4 -vf "select=eq(n\\\\,63)" -vsync 0 -vframes 1 ../workspace/step63.png`, and then open the PNG as an image. Line 0 of the trace names the file for each camera. These are compressed, so a pixel can be off by a shade.\n'
    '  - `episode.json` : whether that episode succeeded, how many steps it took, and exactly which task, scene and seeds it used.\n'
    '  - `benchmark_metrics.json` : extra measurements for that episode, on the setups that produce any. Many produce none, so do not assume this file exists.\n'
    '\n'
    '\n'
    '## Running the simulator yourself\n'
    'You can re-run any recorded episode in the simulator and look at whatever you like. It reads the task, scene and seeds from `episode.json` and the actions from `trace.jsonl`, so it repeats the episode that was recorded. The last section gives the exact command for this setup.\n'
    '\n'
    '\n'
    '## Rules your harness has to follow\n'
    "- While you work you may read anything you need: the recordings, the simulator source, the policy code. Keep to this project and the paths `paths.json` names; nothing else on this machine is part of your task. But the harness you ship must not read the simulator's own answers while an episode is running -- no live simulator state, no true object or goal positions, no success check, no expert actions. In practice that means none of `_check_success`, `sim.data`, `body_xpos`, `site_xpos`, `geom_xpos`. Do not smuggle the answer into the harness.\n"
    '\n'
    '- Do not write a solution for the particular tasks in these recordings. A general rule names no particular task and no particular object, and would behave the same way on a task you have never seen. A lookup does not: a table keyed by task name or id, a coordinate you read off a recording and typed in, a special case for one episode. In particular, a place you move the gripper to must be worked out from this episode\'s own observation -- from what the camera sees, from depth, from the robot\'s own readings -- and never typed in as a number. A constant offset is a different thing and is fine: `nudge(observation, (0, 0, 0.03))` means "lift three centimetres from wherever the gripper is now", which behaves the same on any task.\n'
    '\n'
    '- Do not change how success is decided or how the result is counted. That is not in your file, and reaching for it will not work.\n'
    '\n'
    '\n'
)


ACROSS_EPISODES_SLOT = "<<ACROSS_EPISODES>>"

ACROSS_EPISODES = 'Python values stored on the harness object may or may not survive into the next episode. Clear anything episode-specific in `reset(session_id)`. The one supported way to carry something from one episode to the next is the memory described under "Memory that carries across episodes" below.'


MEMORY_NOTES = (
    "## Memory that carries across episodes\n"
    "Your harness has a memory that survives from one episode to the next.\n"
    "\n"
    "- `tools.remember(key, value)` stores one value under one name. `key` is a string. `value` may be any mixture of numbers, strings, true/false, null, lists and dictionaries whose keys are strings; an array or an image is not one of those. Storing the same name again replaces what was there.\n"
    "- `tools.recall(key)` returns what was stored under that name, or `None` if nothing was.\n"
    "- `tools.memory_keys()` lists the names in use.\n"
    "- `tools.forget(key)` removes one.\n"
    "\n"
    "All the episodes of one evaluation -- one whole pass over the episode set -- share one memory, and it begins empty: the first episodes fill it and the later ones can use it. Episodes run several at a time, so you cannot count on the order in which they write. Nothing carries over into the next evaluation.\n"
    "\n"
    "The memory is saved as `memory.json` beside the recordings, and `../feedback.json` gives the exact paths under `current_harness_memory` and, where there is one, `rejected_revision_memory`. Read it. The recordings tell you what the robot did; this tells you what it remembered, how many times it was read, and how many of those reads found anything.\n"
    "\n"
    "Whether to use the memory at all, what to put in it, and when to read it are yours to decide. What you store follows the same rule as everything else: it must be worked out from what the robot could see, and it must not be an answer written down for a particular task.\n"
)


def tool_facts(setup: Setup) -> list[str]:
    lines: list[str] = []
    served = sorted(setup.available_tools.items())
    missing = sorted(capability for capability, service in setup.tools if service is None)
    if served:
        lines.append("- Models available on this setup: " + ", ".join(f"{capability} = {MODELS[service].model_id}" for capability, service in served) + ". One is loaded only if `HARNESS_CONFIG` lists it and your file mentions it, so name in a `tools.<name>(...)` call, a `tools.has(\"<name>\")` check or one of its request types every model you intend to use. Check with `tools.has(...)` before calling.")
    if missing:
        lines.append("- Never available here (tools.has(...) is False even if you call it, and calling one raises): " + ", ".join(missing) + ".")
    if "grasp" in setup.available_tools:
        lines.append("- The grasp model: give it a picture, that picture's depth, the camera's lens matrix and pose, the camera's optical convention, and a mask of the object, and it returns ranked six-degree-of-freedom grasp poses in the same coordinates the gripper is reported in -- `tools.grasp(GraspRequest(rgb, depth_m, intrinsics, camera_to_world, optical_convention, mask))` -> `GraspResult(.candidates)`, each `GraspCandidate(.pose_world` 4x4`, .score, .width_m)`. Get the mask from the segmenter. It needs depth, and it was trained for this robot's gripper. A pose is where and how to hold the object; moving there is still your job, and on a setup whose action is a change from the current pose the ready-made movement commands will not turn the wrist, so you would have to write the rotation numbers yourself.")
    return lines


def setup_facts(setup: Setup) -> list[str]:
    lines: list[str] = []
    cameras = setup.environment.cameras
    with_depth = [item for item in cameras if item.has_depth]
    camera_list = ", ".join(f"{item.name} ({item.width}x{item.height})" for item in cameras)
    if with_depth:
        upright = "the BOTTOM" if cameras[0].optical_convention == "opengl_rub" else "the TOP"
        lines.append(f"- Cameras: {camera_list}. Depth is on for: " + ", ".join(item.name for item in with_depth) + f". Those cameras carry .depth_m (metres, one value per colour pixel), .depth_valid, .intrinsics and .camera_to_world. Use roboarchitect.harness.geometry rather than projecting by hand -- here row 0 of the stored picture is {upright} of the scene, and the module already accounts for that.")
    else:
        lines.append(f"- Cameras: {camera_list}. There is no depth here: every camera's .depth_m, .intrinsics and .camera_to_world are None, so anything needing a distance in metres cannot work.")
    lines.append(f"- The robot is a {setup.environment.embodiment}.")
    spec = setup.policy.action_spec
    absolute = "absolute" in spec.channel_semantics
    lines.append(
        f"- The action the policy returns, and the action you must return: {len(spec.channel_names)} numbers -- " + ", ".join(spec.channel_names) + ". "
        + ("They are a target pose, not a change from where the gripper is now. " if absolute else "The movement numbers are a change from where the gripper is now, not a destination. ")
        + f"Rotation is {spec.rotation_representation.replace('_', '-')}. "
        + (f"Every value must be between -1 and +1, and one unit means the matching per-channel scale, here {list(spec.controller_output_scale)}, so the first number at 1.0 commands {spec.controller_output_scale[0]} in its own unit. " if spec.value_encoding == "normalized_controller" else "The values are physical, in the units the channels name, and are not clipped to a range. ")
        + ("For the gripper, +1 is closed and -1 is open. " if spec.gripper_convention in ("closed_positive", "binary_closed_one") else "For the gripper, +1 is open and -1 is closed. " if spec.gripper_convention in ("open_positive", "binary_open_one") else "")
        + f"Exactly {setup.policy.execution_count} action(s) execute per step and a chunk may hold at most {setup.policy.chunk_horizon}."
    )
    eef_frames = sorted(
        {item.reference_frame for item in setup.environment.robot_state if item.quantity == "end_effector_pose"}
    )
    if eef_frames:
        lines.append("- The gripper position in observation.proprioception is given in the " + ("same coordinates as the points geometry.pixel_to_world returns and the targets harness.motion takes, so you can compare them directly." if with_depth else f"'{eef_frames[0]}' frame, which is also the frame harness.motion takes targets in. There is no depth here, so a target has to come from proprioception or from your own reasoning about the scene, not from geometry.pixel_to_world, which returns None."))
    controller = make_controller(spec)
    if controller is None:
        lines.append("- Movement commands: roboarchitect.harness.motion does NOT support this action layout, so make_controller(spec) returns None here. Either use the policy's action or compute every number yourself.")
    else:
        lines.append(f"- Movement commands: roboarchitect.harness.motion can drive this setup. Its layout is named '{controller.layout}'; grep that name in harness/motion.py to see exactly how a move is computed. make_controller(chunk.spec) gives you move_to, nudge, hold and set_gripper. You may return one of those instead of the policy's action on any step, and work every destination out from the observation.")
        if not controller.is_delta:
            note = "- Here the action IS a target pose, not a change, so a movement command has to supply a rotation as well. Call controller.note(chunk) with the policy's own chunk each step and it will reuse that rotation."
            if setup.environment.benchmark == "simpler":
                note += f" Do not rely on the rotation the controller works out from the gripper pose here: this setup labels its rotation numbers '{spec.rotation_representation}', but the controller underneath reads those three numbers as a rotation VECTOR."
            lines.append(note)
    if setup.policy.server == "xvla":
        lines.append("- This setup's policy reads VLARequest.context: passing `policy_resample_index=<n>` in the context tuple makes it draw a different action for the same picture.")
    return lines


def experiment_facts(setups: Mapping[str, Setup], primary: str, metric: str, gpu_ids: tuple[int, ...]) -> str:
    lines = tool_facts(setups[primary])
    blocks: dict[tuple[str, ...], list[str]] = {}
    for name in sorted(setups):
        blocks.setdefault(tuple(setup_facts(setups[name])), []).append(name)
    if len(blocks) == 1:
        lines.extend(next(iter(blocks)))
    else:
        lines.append(f"- This experiment covers more than one robot setup, and every episode arrives through the same file. There are {len(blocks)} of them, described one after another below. Tell them apart at runtime from what the observation and the action specification actually say, and never from anything you hardcode about which job is running.")
        for index, block in enumerate(blocks, 1):
            lines.append(f"### Setup {index} of {len(blocks)}")
            lines.extend(block)
    command = f"PYTHONPATH={CODE_ROOT} {sys.executable} -m roboarchitect.replay --episode <an episode folder> --out ../workspace/replay --render-gpu {gpu_ids[-1]} --steps 40 --depth"
    pool = ", ".join(str(item) for item in gpu_ids)
    lines.append(f"- To re-run an episode in the simulator yourself, this is the command for this setup. Fill in the episode folder from `../feedback.json`:\n\n  ```\n  {command}\n  ```\n\n  It writes one PNG per camera per step, optionally the depth as `.npy`, and a `replay.json` saying whether the episode succeeded. `--steps` stops early; leave it out for the whole episode. Point `--actions` at a different `trace.jsonl` to run a DIFFERENT action sequence in the same scene, which is how you check what would have happened if the robot had done something else. Add `--help` for the rest. Starting the simulator takes about half a minute, and a hundred steps takes about as long again.\n\n  This experiment was given graphics card(s) {pool} and nothing else on this machine. Leave `--render-gpu {gpu_ids[-1]}` as it is written above. Do not put another number there, do not set CUDA_VISIBLE_DEVICES to another card, and do not start any other program on another card.")
    lines.append(f"- How the per-episode outcomes become this setup's single number is in benchmarks/metrics.py, under `{metric}`. Read it: it decides whether a gain on one task can pay for a loss on another.")
    return "\n\n".join(lines)


def revision_prompt(notes: str) -> str:
    base = REVISION_PROMPT.replace(ACROSS_EPISODES_SLOT, ACROSS_EPISODES)
    notes_section = f"## Facts about THIS setup\n{notes.strip()}\n" if notes.strip() else ""
    return base + f"{MEMORY_NOTES.strip()}\n\n\n" + notes_section
