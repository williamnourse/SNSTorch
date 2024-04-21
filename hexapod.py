import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import time
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron, SpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse, SpikingSynapse
from sns_toolbox.renderer import render
from sns_toolbox.plot_utilities import spike_raster_plot

"""Linear remapping"""
def remap(data, start, end):
    return (1/(end-start))*(data-start)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mujoco Model Setup
"""
XML=r"""
<mujoco model="hexapod">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" fromto="-1.0 0.0 0.0 1.0 0.0 0.0" size="0.15" type="capsule" density="100"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="-0.9 0 0">
        <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux_0_geom" size="0.08" type="capsule"/>
        <body name="aux_0" pos="0.0 -0.2 0">
          <joint axis="0 0 -1" name="hip_0" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.0 -0.3 0.0" name="front_left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.0 -0.3 0">
            <joint axis="-1 0 0" name="knee_0" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.0 -0.5 0.0" name="front_left_ankle_geom" size="0.08" type="capsule"/>
            <site name="foot_0" type="capsule" fromto="0.0 0.0 0.0 0.0 -0.5 0.0" rgba="0.8 0.6 0.4 0" size="0.1"/>
          </body>
        </body>
      </body>
      <body name="center_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="0.0 -0.2 0">
          <joint axis="0 0 -1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.0 -0.3 0.0" name="center_left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.0 -0.3 0">
            <joint axis="-1 0 0" name="knee_1" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.0 -0.5 0.0" name="center_left_ankle_geom" size="0.08" type="capsule"/>
            <site name="foot_1" type="capsule" fromto="0.0 0.0 0.0 0.0 -0.5 0.0" rgba="0.8 0.6 0.4 0" size="0.1"/>
          </body>
        </body>
      </body>
      <body name="back_left_leg" pos="0.9 0 0">
        <geom fromto="0.0 0.0 0.0 0.0 -0.2 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="0.0 -0.2 0">
          <joint axis="0 0 -1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.0 -0.3 0.0" name="back_left_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.0 -0.3 0">
            <joint axis="-1 0 0" name="knee_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.0 -0.5 0.0" name="back_left_ankle_geom" size="0.08" type="capsule"/>
            <site name="foot_2" type="capsule" fromto="0.0 0.0 0.0 0.0 -0.5 0.0" rgba="0.8 0.6 0.4 0" size="0.1"/>
          </body>
        </body>
      </body>
      <body name="back_right_leg" pos="0.9 0 0">
        <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="0.0 0.2 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.0 0.3 0.0" name="back_right_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.0 0.3 0">
            <joint axis="1 0 0" name="knee_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.0 0.5 0.0" name="back_right_ankle_geom" size="0.08" type="capsule"/>
            <site name="foot_3" type="capsule" fromto="0.0 0.0 0.0 0.0 0.5 0.0" size="0.1" rgba="0.8 0.6 0.4 0"/>
          </body>
        </body>
      </body>
      <body name="center_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="0.0 0.2 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.0 0.3 0.0" name="center_right_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.0 0.3 0">
            <joint axis="1 0 0" name="knee_4" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.0 0.5 0.0" name="center_right_ankle_geom" size="0.08" type="capsule"/>
            <site  name="foot_4" type="capsule" fromto="0.0 0.0 0.0 0.0 0.5 0.0" rgba="0.8 0.6 0.4 0" size="0.1"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="-0.9 0 0">
        <geom fromto="0.0 0.0 0.0 0.0 0.2 0.0" name="aux_5_geom" size="0.08" type="capsule"/>
        <body name="aux_5" pos="0.0 0.2 0">
          <joint axis="0 0 1" name="hip_5" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 0.0 0.3 0.0" name="front_right_leg_geom" size="0.08" type="capsule"/>
          <body pos="0.0 0.3 0">
            <joint axis="1 0 0" name="knee_5" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 0.0 0.5 0.0" name="front_right_ankle_geom" size="0.08" type="capsule"/>
            <site name="foot_5" type="capsule" fromto="0.0 0.0 0.0 0.0 0.5 0.0" rgba="0.8 0.6 0.4 0" size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_0" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_0" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_5" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="knee_5" gear="150"/>
  </actuator>
  <sensor>
    <touch site="foot_0"/>
    <touch site="foot_1"/>
    <touch site="foot_2"/>
    <touch site="foot_3"/>
    <touch site="foot_4"/>
    <touch site="foot_5"/>
  </sensor>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML,{})
data = mujoco.MjData(model)
# for i in range(len(data.qpos)):
#   print(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
hip_0_id = model.joint('hip_0').id+6
hip_1_id = model.joint('hip_1').id+6
hip_2_id = model.joint('hip_2').id+6
hip_3_id = model.joint('hip_3').id+6
hip_4_id = model.joint('hip_4').id+6
hip_5_id = model.joint('hip_5').id+6
knee_0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_0')+6
knee_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_1')+6
knee_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_2')+6
knee_3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_3')+6
knee_4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_4')+6
knee_5_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'knee_5')+6
hip_angles = [hip_0_id, hip_1_id, hip_2_id, hip_3_id, hip_4_id, hip_5_id]
knee_angles = [knee_0_id, knee_1_id, knee_2_id, knee_3_id, knee_4_id, knee_5_id]

motor_ids_hip = [0, 2, 4, 6, 8, 10]
motor_ids_knee = [1, 3, 5, 7, 9, 11]

foot_0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'front_left_ankle_geom')
foot_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'center_left_ankle_geom')
foot_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'back_left_ankle_geom')
foot_3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'back_right_ankle_geom')
foot_4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'center_right_ankle_geom')
foot_5_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'front_right_ankle_geom')
foot_ids = [foot_0_id, foot_1_id, foot_2_id, foot_3_id, foot_4_id, foot_5_id]

angles_hip_lower = np.zeros(6) - 0.5265
angles_hip_upper = np.zeros(6) + 0.5265
angles_knee_lower = np.zeros(6) - 1.2247
angles_knee_upper = np.zeros(6) - 0.5206

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Joint Angle Controller
"""
reversal_ex = 5.0
reversal_in = -2.0
reversal_mod = 0.0
nrn_nonspiking = NonSpikingNeuron()

net_angle = Network(name='Joint Angle Controller')

net_angle.add_neuron(nrn_nonspiking, name='theta_actual')
net_angle.add_neuron(nrn_nonspiking, name='theta_target')
net_angle.add_input('theta_actual')
# net_angle.add_input('theta_target')
# net_angle.add_output('theta_actual')
# net_angle.add_output('theta_target')

net_angle.add_neuron(nrn_nonspiking, name='real_greater_target')
net_angle.add_neuron(nrn_nonspiking, name='real_less_target')
# net_angle.add_output('real_greater_target')
# net_angle.add_output('real_less_target')
g_add = 1/(reversal_ex-1)
g_sub = -(g_add*reversal_ex)/reversal_in
syn_add = NonSpikingSynapse(max_conductance=g_add, reversal_potential=reversal_ex, e_lo=0.0, e_hi=1.0)
syn_sub = NonSpikingSynapse(max_conductance=g_sub, reversal_potential=reversal_in, e_lo=0.0, e_hi=1.0)
net_angle.add_connection(syn_add, 'theta_actual', 'real_greater_target')
net_angle.add_connection(syn_sub, 'theta_target', 'real_greater_target')
net_angle.add_connection(syn_sub, 'theta_actual', 'real_less_target')
net_angle.add_connection(syn_add, 'theta_target', 'real_less_target')

net_angle.add_neuron(nrn_nonspiking, name='torque_more')
net_angle.add_neuron(nrn_nonspiking, name='torque_less')
k_more = 0.5
k_less = 0.5
g_torque_more = k_more/(reversal_ex-k_more)
g_torque_less = k_less/(reversal_ex-k_less)
syn_torque_more = NonSpikingSynapse(max_conductance=g_torque_more, reversal_potential=reversal_ex, e_lo=0.0, e_hi=1.0)
syn_torque_less = NonSpikingSynapse(max_conductance=g_torque_less, reversal_potential=reversal_ex, e_lo=0.0, e_hi=1.0)
net_angle.add_connection(syn_torque_more,'real_greater_target', 'torque_less')
net_angle.add_connection(syn_torque_less,'real_less_target', 'torque_more')
net_angle.add_output('torque_more')
net_angle.add_output('torque_less')

# render(net_angle)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Joint Angle Simulation Loop
"""
# sns_leg = net_angle.compile(dt=0.1, backend='numpy')
# model.opt.timestep = 0.0001
# with mujoco.viewer.launch_passive(model, data) as viewer:
#   # Close the viewer automatically after 30 wall-seconds.
#   start = time.time()
#   i = 0
#   dir = 1
#   inputs = np.zeros(2)
#   torque = 0
#   ref = []
#   num_steps = 20000
#   output = np.zeros([num_steps, 2])
#   theta = []
#   while viewer.is_running() and i<num_steps:
#     step_start = time.time()
#
#     ref_angle = 0.5*np.sin(0.001*i)+0.5
#     ref.append(ref_angle)
#     print(ref_angle)
#
#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     if i % 100 == 0:
#       dir *= -1
#     data.ctrl[motor_ids_hip] = -1
#     data.ctrl[motor_ids_knee] = -1
#     data.ctrl[motor_ids_knee[0]] = torque
#
#     mujoco.mj_step(model, data)
#
#     angles_hip = data.qpos[hip_angles]
#     angles_hip_map = remap(angles_hip, angles_hip_lower, angles_hip_upper)
#     angles_knee = data.qpos[knee_angles]
#     angles_knee_map = remap(angles_knee, angles_knee_lower, angles_knee_upper)
#
#     inputs[0] = angles_knee_map[0]
#     inputs[1] = ref_angle
#     output[i,:] = sns_leg(inputs)
#     theta.append(inputs[0])
#     torque = output[i,0] - output[i,1]
#
#     print(torque)
#     # print('Hip Angles:')
#     # print(angles_hip_map)
#     # print('Knee Angles:')
#     # print(angles_knee_map)
#     # print('Sensor Readings:')
#     # print(data.sensordata)
#
#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()
#
#     # Rudimentary time keeping, will drift relative to wall clock.
#     # time_until_next_step = model.opt.timestep - (time.time() - step_start)
#     # if time_until_next_step > 0:
#     #   time.sleep(time_until_next_step)
#     i += 1
# output = output.transpose()
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(theta, label='Theta')
# # plt.plot(output[0,:], label='Actual Nrn')
# plt.plot(ref, label='Reference')
# # plt.plot(output[1,:], label='Target Nrn')
# plt.legend()
# plt.subplot(2,1,2)
# # plt.plot(output[2,:], label='Actual > Target Nrn')
# # plt.plot(output[3,:], label='Actual < Target Nrn')
# plt.plot(output[0,:], label='More Nrn')
# plt.plot(output[1,:], label='Less Nrn')
# plt.legend()

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CPG Network
"""
def add_cpg(net, suffix, start_swing=True):
    if start_swing:
        swing_initial = 0.5
        stance_initial = 0.0
    else:
        swing_initial = 0.0
        stance_initial = 0.5
    hc_bias = 0.1
    nrn_hc = NonSpikingNeuron(bias=hc_bias)
    nrn_slow = NonSpikingNeuron(membrane_capacitance=100)
    net.add_neuron(nrn_hc, name='Swing HC'+suffix, initial_value=swing_initial)
    net.add_neuron(nrn_slow, name='Swing Slow'+suffix)
    net.add_neuron(nrn_hc, name='Stance HC'+suffix, initial_value=stance_initial)
    net.add_neuron(nrn_slow, name='Stance Slow'+suffix)

    g_self = 0.5
    a = 0.9
    g_sf = (g_self * (reversal_ex - a) - a) / (a - reversal_in)
    syn_cpg_self = NonSpikingSynapse(max_conductance=g_self, reversal_potential=reversal_ex, e_hi=1)
    syn_cpg_ex_slow = NonSpikingSynapse(max_conductance=1 / (reversal_ex - 1), reversal_potential=reversal_ex, e_hi=1)
    syn_cpg_slow_in = NonSpikingSynapse(max_conductance=g_sf, reversal_potential=reversal_in, e_hi=1)
    g_in_hc = -hc_bias / reversal_in
    syn_cpg_hc = NonSpikingSynapse(max_conductance=g_in_hc, reversal_potential=reversal_in, e_hi=1)

    net.add_connection(syn_cpg_self, 'Swing HC'+suffix, 'Swing HC'+suffix)
    net.add_connection(syn_cpg_ex_slow, 'Swing HC'+suffix, 'Swing Slow'+suffix)
    net.add_connection(syn_cpg_slow_in, 'Swing Slow'+suffix, 'Swing HC'+suffix)
    net.add_connection(syn_cpg_self, 'Stance HC'+suffix, 'Stance HC'+suffix)
    net.add_connection(syn_cpg_ex_slow, 'Stance HC'+suffix, 'Stance Slow'+suffix)
    net.add_connection(syn_cpg_slow_in, 'Stance Slow'+suffix, 'Stance HC'+suffix)
    net.add_connection(syn_cpg_hc, 'Swing HC'+suffix, 'Stance HC'+suffix)
    net.add_connection(syn_cpg_hc, 'Stance HC'+suffix, 'Swing HC'+suffix)

net_cpg = Network(name='CPG Joint Controller')
add_cpg(net_cpg, '')
net_cpg.add_output('Swing HC')
net_cpg.add_output('Stance HC')

# render(net_cpg)

sns_cpg = net_cpg.compile(dt=0.1, backend='numpy')
num_steps = 20000
outputs = np.zeros([num_steps, 2])
for i in range(num_steps):
    outputs[i,:] = sns_cpg()
outputs = outputs.transpose()
# plt.figure()
# plt.plot(outputs[0,:])
# plt.plot(outputs[1,:])
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Leg Controller
"""
def add_leg(net, suffix, start_swing):
    add_cpg(net, ' h'+suffix, start_swing=start_swing)
    net.add_network(net_angle, suffix=' h'+suffix)
    add_cpg(net, ' k'+suffix, start_swing=start_swing)
    net.add_network(net_angle, suffix=' k'+suffix)
    # net.add_input('Swing HC k')
    net.add_connection(syn_add, 'Swing HC h'+suffix, 'theta_target h'+suffix)
    net.add_connection(syn_add, 'Swing HC k'+suffix, 'theta_target k'+suffix)
    net.add_connection(syn_add, 'Swing HC k'+suffix, 'Swing HC h'+suffix)
    net.add_connection(syn_add, 'Stance HC h'+suffix, 'Stance HC k'+suffix)
    # net.add_output('Swing HC h'+suffix)
    # net.add_output('Stance HC h'+suffix)
    # net.add_output('Swing HC k'+suffix)
    # net.add_output('Stance HC k'+suffix)


net_leg = Network(name='Leg Controller')
add_leg(net_leg, '0', True)

sns_leg = net_leg.compile(dt=0.1, backend='numpy')

# model.opt.timestep = 0.0001
# with mujoco.viewer.launch_passive(model, data) as viewer:
#   # Close the viewer automatically after 30 wall-seconds.
#   start = time.time()
#   i = 0
#   inputs = np.zeros(2)
#   torque_knee = 0
#   torque_hip = 0
#   num_steps = 50000
#   output = np.zeros([num_steps, 4])
#   while viewer.is_running() and i<num_steps:
#     step_start = time.time()
#
#
#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     data.ctrl[motor_ids_hip] = -1
#     data.ctrl[motor_ids_knee] = -1
#     data.ctrl[motor_ids_hip[0]] = torque_hip
#     data.ctrl[motor_ids_knee[0]] = torque_knee
#
#     mujoco.mj_step(model, data)
#
#     angles_hip = data.qpos[hip_angles]
#     angles_hip_map = remap(angles_hip, angles_hip_lower, angles_hip_upper)
#     angles_knee = data.qpos[knee_angles]
#     angles_knee_map = remap(angles_knee, angles_knee_lower, angles_knee_upper)
#
#     inputs[0] = angles_hip_map[0]
#     inputs[1] = angles_knee_map[0]
#     output[i,:] = sns_leg(inputs)
#     torque_hip = output[i,0] - output[i,1]
#     torque_knee = output[i,2] - output[i,3]
#
#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()
#
#     # Rudimentary time keeping, will drift relative to wall clock.
#     # time_until_next_step = model.opt.timestep - (time.time() - step_start)
#     # if time_until_next_step > 0:
#     #   time.sleep(time_until_next_step)
#     i += 1
# output = output.transpose()
# plt.figure()
# plt.plot(output[0,:], label='H More Torque')
# plt.plot(output[1,:], label='H Less Torque')
# plt.plot(output[2,:], label='K More Torque')
# plt.plot(output[3,:], label='K Less Torque')
# plt.legend()

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Walking Controller
"""
net_walk = Network(name='Walking Controller')
add_leg(net_walk, '0', True)
add_leg(net_walk, '1', False)
add_leg(net_walk, '2', True)
add_leg(net_walk, '3', False)
add_leg(net_walk, '4', True)
add_leg(net_walk, '5', False)

sns_walk = net_walk.compile(dt=0.1, backend='numpy')
model.opt.timestep = 0.0001
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  i = 0
  inputs = np.zeros(2*6)
  torque_knee = np.zeros(6)-1
  torque_hip = np.zeros(6)-1
  num_steps = 50000
  output = np.zeros([num_steps, 4*6])
  while viewer.is_running() and i<num_steps:
    step_start = time.time()


    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    # data.ctrl[motor_ids_hip] = -1
    # data.ctrl[motor_ids_knee] = -1
    data.ctrl[motor_ids_hip] = torque_hip
    data.ctrl[motor_ids_knee] = torque_knee

    mujoco.mj_step(model, data)

    angles_hip = data.qpos[hip_angles]
    angles_hip_map = remap(angles_hip, angles_hip_lower, angles_hip_upper)
    angles_knee = data.qpos[knee_angles]
    angles_knee_map = remap(angles_knee, angles_knee_lower, angles_knee_upper)

    inputs[0::2] = angles_hip_map
    inputs[1::2] = angles_knee_map
    output[i,:] = sns_walk(inputs)
    torque_hip = output[i,0::4] - output[i,1::4]
    torque_knee = output[i,2::4] - output[i,3::4]

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    # time_until_next_step = model.opt.timestep - (time.time() - step_start)
    # if time_until_next_step > 0:
    #   time.sleep(time_until_next_step)
    i += 1

plt.show()