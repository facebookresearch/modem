if _ROBOT_VIZ:
    from vtils.plotting.srv import *

    timing_SRV = SRV(legends=("timeing",))
    timing_SRV_t0 = time.time()

    def configure_robot_viz(self, robot_config):
        for name, device in robot_config.items():
            if device["sensor_names"]:
                device["sensor_viz"] = SRV(
                    legends=tuple(device["sensor_names"]), fig_name="Sensor: " + name
                )
            if device["actuator_names"]:
                device["actuator_viz"] = SRV(
                    legends=tuple(device["actuator_names"]),
                    fig_name="Actuator: " + name,
                )

    def update_robot_viz(self, update_sensor=False, update_control=False):
        for name, device in self.robot_config.items():
            if device["sensor_names"] and update_sensor:
                device["sensor_viz"].append(
                    x_data=device["sensor_time"] * np.ones_like(device["sensor_data"]),
                    y_data=device["sensor_data"],
                )
            if device["actuator_names"] and update_control:
                device["controls"] = np.asarray(device["controls"])
                device["actuator_viz"].append(
                    x_data=device["sensor_time"] * np.ones_like(device["controls"]),
                    y_data=device["controls"],
                )

    def clear_robot_viz(self, clear_sensor=False, clear_control=False):
        for name, device in self.robot_config.items():
            if clear_sensor and device["sensor_names"]:
                device["sensor_viz"].clear()
            if clear_control and device["actuator_names"]:
                device["actuator_viz"].clear()
