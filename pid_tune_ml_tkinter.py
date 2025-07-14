import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import differential_evolution
from dataclasses import dataclass
import time

### Make timestep changable, to match scan time on PLC PID

# Constants
settling_tolerance = 0.005

# points_per_second = params.points_per_second
# update_rate = 20
# points_per_second = 50
# pps=1/points_per_second
# sec=points_per_second/1000
# sim_time = 20
# num_points = int(sim_time * points_per_second)
# t_eval = np.linspace(0, sim_time, num_points)
# timestep = t_eval[1] - t_eval[0]
# t_eval = np.insert(t_eval, 0, 0.0)
# t_eval = np.linspace(0, sim_time, num_points)
# timestep = t_eval[1] - t_eval[0]

@dataclass
class SimulationParams:
    Kp: float
    Ki: float
    Kd: float
    cv_min: float
    cv_max: float
    cv_start: float
    pv_min: float
    pv_max: float
    pv_start: float
    pv_final: float
    process_gain: float
    dead_time: float
    time_constant: float
    setpoint: float
    timestep: float

# PID Controller Class
class PID:
    # Initialize PID, set parameters and set integral and error to zero
    def __init__(self, Kp, Ki, Kd, cv_start):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = cv_start / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0

    # Reset PID, set integral and error to zero
    def reset(self, cv_start):
        self.integral = cv_start / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0
        
    # Core PID logic  
    def update(self, error, dt, cv_min, cv_max):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        control = np.clip(self.Kp * error + self.Ki * self.integral + self.Kd * derivative, cv_min, cv_max)
        self.prev_error = error
        integral = self.integral
        return control, integral


# System Simulation
def simulate_system(params: SimulationParams):
    # Call the PID and make sure it's reset at the start
    pid = PID(params.Kp, params.Ki, params.Kd, params.cv_start)
    pid.reset(params.cv_start)
    
    # Initialize CV and PV based on setpoints
    pv = np.clip(params.pv_start, params.pv_min, params.pv_max)
    sp = params.setpoint
    cv = params.cv_start
    
    points_per_second = 1 / params.timestep
    #points_per_second = 40
    sim_time = 20
    num_points = int(sim_time * points_per_second)
    t_eval = np.linspace(0, sim_time, num_points)
#     t_eval = np.insert(t_eval, 0, 0.0)
    #t_eval = np.arange(0, sim_time+params.timestep, params.timestep)

    # Fill out the cv buffer with the CV init value to avoid starting at zero
    cv_delay_steps = max(1, int(np.round(params.dead_time / params.timestep)))
    cv_buffer = [params.cv_start] * cv_delay_steps
    disturbance_magnitude = 0.001
    disturbance_time = 5
    disturbance_duration = 3.0
    disturbance_end = disturbance_time + disturbance_duration
    disturbance_value = params.pv_final * disturbance_magnitude
    
    # Initialize lists for cv/pv/sp histories
    pv_array = [pv]
    cv_history = [cv]
    sp_array = [sp]
    
    # Run the actual simulation for x time
    for i, t in enumerate(t_eval[1:]):
        # Update the PID every cycle
        error = sp - pv
        cv, integral = pid.update(error, params.timestep, params.cv_min, params.cv_max)
        
        cv_buffer.append(cv)
        delayed_cv = cv_buffer.pop(0)
        # Update the PID simulation
        dpv = (-pv + params.process_gain * delayed_cv) / params.time_constant
        pv += dpv * params.timestep
        pv = np.clip(pv, params.pv_min, params.pv_max)
        
        # Add the PV, cv and SP to arrays for plotting
        pv_array.append(pv)
        cv_history.append(cv)
        sp_array.append(sp)
    return np.array(pv_array), np.array(sp_array), np.array(cv_history), np.array(t_eval)

# Compute Settling Time
def compute_settling_time(pv, t, pv_final, tolerance=settling_tolerance):
    pv = np.asarray(pv)
    t = np.asarray(t)

    min_len = min(len(pv), len(t))
    pv = pv[:min_len]
    t = t[:min_len]

    band_upper = pv_final * (1 + tolerance)
    band_lower = pv_final * (1 - tolerance)

    for i in range(min_len):
        window = pv[i:]
        if np.all((window >= band_lower) & (window <= band_upper)):
            return t[i]
    return t[-1]  # If it never settles

def truncate_float(value, decimals=2):
    factor = 10 ** decimals
    return int(value * factor) / factor

trial_counter = 0

# GUI application
class PIDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PID Simulation with Optimization")
        self.geometry("1215x600")
        self.kp_var = tk.DoubleVar(value=0.185)
        self.ki_var = tk.DoubleVar(value=0.7)
        self.kd_var = tk.DoubleVar(value=0.0)
        self.cv_min = tk.DoubleVar(value=0.0)
        self.cv_max = tk.DoubleVar(value=100.0)
        self.cv_start = tk.DoubleVar(value=30.0)
        self.cv_final = tk.DoubleVar(value=70.0)
        self.pv_min = tk.DoubleVar(value=0.0)
        self.pv_max = tk.DoubleVar(value=250.0)
        self.pv_start = tk.DoubleVar(value=10.0)
        self.pv_final = tk.DoubleVar(value=45.0)
        self.dead_time = tk.DoubleVar(value=0.1)
        self.tau = tk.DoubleVar(value=0.75)
        self.pid_update = tk.DoubleVar(value=20)
        self.sixtythreepctvalue = tk.DoubleVar(value=self.compute_sixtythree_pct_value())
        self.time_constant = tk.DoubleVar(value=self.compute_time_constant())
        self.truncated_time_constant = tk.StringVar(value=str(truncate_float(self.time_constant.get(), 3)))
        self.process_gain = tk.DoubleVar(value=self.compute_process_gain())
        self.max_overshoot = tk.StringVar(value="0.0")
        self.max_overshoot_pct = tk.StringVar(value="0.0%")
        self.settling_time = tk.StringVar(value="0.0s")
        self.elapsed_time = tk.StringVar(value="0.0s")
        self.setpoint = tk.DoubleVar(value=self.pv_final.get())
        
        #Add logic for live updates of necessary values
        self.live_process_gain = tk.StringVar()
        self.live_sixtythree_pct = tk.StringVar()
        self.live_time_constant = tk.StringVar()
        # These variables affect process gain
        self.cv_start.trace_add("write", lambda *args: self.update_simulation_outputs())
        self.cv_final.trace_add("write", lambda *args: self.update_simulation_outputs())
        self.pv_start.trace_add("write", lambda *args: self.update_simulation_outputs())
        self.pv_final.trace_add("write", lambda *args: self.update_simulation_outputs())

        # These affect time constant and 63.2% value
        self.tau.trace_add("write", lambda *args: self.update_simulation_outputs())
        self.dead_time.trace_add("write", lambda *args: self.update_simulation_outputs())
#         self.disturbance = tk.BooleanVar(value=True)
#         self.noise = tk.BooleanVar(value=False)
        self.update_simulation_outputs()
        self.create_widgets()
        self.init_plot()

    def create_widgets(self):     
        process = ttk.LabelFrame(self, text="Process")
        process.grid(row=0, column=0, sticky="nwns", padx=5, pady=5)
        process_row = 0
        ttk.Label(process, text="Process Step").grid(row=process_row, column=0, stick="e")
        
        process_row +=1
        ttk.Label(process, text="CV Start").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_start).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="CV Final").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_final).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="PV Start").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_start).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="PV Final").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_final).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="Process Limits").grid(row=process_row, column=0, stick="e")
        
        process_row +=1
        ttk.Label(process, text="CV Min").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_min).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="CV Max").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_max).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="PV Min").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_min).grid(row=process_row, column=1)
        
        process_row +=1
        ttk.Label(process, text="PV Max").grid(row=process_row, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_max).grid(row=process_row, column=1)
        
        simulation = ttk.LabelFrame(self, text="Simulation")
        simulation.grid(row=0, column=1, sticky="nwns", padx=5, pady=5)
        
        simulation_row = 0
        ttk.Label(simulation, text="Process Parameters").grid(row=simulation_row, column=0, sticky="w")
        
        simulation_row += 1
        ttk.Label(simulation, text="Process Gain(Kp):").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.live_process_gain, state="readonly").grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Dead Time (θp):").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.dead_time).grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="63.2% Value (t0.632):").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.live_sixtythree_pct, state="readonly").grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Time at 63.2% Value (tau):").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.tau).grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Time Constant(τp):").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.live_time_constant, state="readonly").grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="PID Update Rate(ms):").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.pid_update).grid(row=simulation_row, column=1)

#         ttk.Checkbutton(simulation, text="Disturbance", variable=self.disturbance).grid(row=3, column=0, sticky="w")
#         ttk.Checkbutton(simulation, text="Measurement Noise", variable=self.noise).grid(row=4, column=0, sticky="w")
        
        simulation_row += 1
        ttk.Button(simulation, text="Optimize PID", command=self.run_optimization).grid(row=simulation_row, column=0, columnspan=2, pady=5)
        
        simulation_row += 1
        ttk.Label(simulation, text="PID Tuning Parmeters").grid(row=simulation_row, column=0, sticky="w")
        
        simulation_row += 1
        ttk.Label(simulation, text="Kp:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.kp_var).grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Ki:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.ki_var).grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Kd:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.kd_var).grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Setpoint:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.setpoint).grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Button(simulation, text="Manual Simulation", command=self.run_manual).grid(row=simulation_row, column=0, columnspan=2, pady=5)
        
        simulation_row += 1
        ttk.Label(simulation, text="Max Overshoot:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.max_overshoot, state="readonly").grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Max Overshoot Percent:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.max_overshoot_pct, state="readonly").grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Settling Time:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.settling_time, state="readonly").grid(row=simulation_row, column=1)
        
        simulation_row += 1
        ttk.Label(simulation, text="Elapsed Time:").grid(row=simulation_row, column=0, sticky="e")
        ttk.Entry(simulation, textvariable=self.elapsed_time, state="readonly").grid(row=simulation_row, column=1)
        
        instruction = ttk.LabelFrame(self, text="Instructions")
        instruction.grid(row=1, column=0, columnspan=2, sticky="nwew", padx=5, pady=5)
        
        #ttk.Label(instruction, text="1. Make a step change in your CV.  This change should be in the normal operating range of your process, and should be large enough to see a noticable(>10% of full range) change in your PV").grid(row=0, column=0, stick="e")

        self.plot_frame = ttk.LabelFrame(self, text="Live Plot")
        self.plot_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.ax.set_title("Live PID Optimization")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.pv_line, = self.ax.plot([], [], label="PV")
        self.sp_line, = self.ax.plot([], [], 'r--', label="SP")
        self.cv_line, = self.ax.plot([], [], 'g:', label="CV")
        self.ax.legend()
        self.fig.tight_layout()
        
    def update_plot(self, t, pv, sp, cv, title):
        self.ax.set_title(title)
        self.pv_line.set_data(t, pv)
        self.sp_line.set_data(t, sp)
        self.cv_line.set_data(t, cv)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()

    def compute_process_gain(self):
        try:
            pv_final = float(self.pv_final.get())
            pv_start = float(self.pv_start.get())
            cv_final = float(self.cv_final.get())
            cv_start = float(self.cv_start.get())
            if cv_final == cv_start:
                return 0.0  # avoid divide by zero
            return (pv_final - pv_start) / (cv_final - cv_start)
        except (tk.TclError, ValueError):
            return 0.0

    def compute_time_constant(self):
        try:
            tau = float(self.tau.get())
            dead_time = float(self.dead_time.get())
            return max(1.1 * (tau - dead_time), 0.1)
        except (tk.TclError, ValueError):
            return 0.1

    def compute_sixtythree_pct_value(self):
        try:
            pv_final = float(self.pv_final.get())
            pv_start = float(self.pv_start.get())
            return (abs(pv_final - pv_start) * 0.632) + pv_start
        except (tk.TclError, ValueError):
            return 0.0
    
    def update_simulation_outputs(self):
        pg = self.compute_process_gain()
        sixtythree = self.compute_sixtythree_pct_value()
        taup = self.compute_time_constant()
        
        self.process_gain.set(pg)
        self.time_constant.set(taup)

        self.live_process_gain.set(str(truncate_float(pg, 3)))
        self.live_sixtythree_pct.set(str(truncate_float(sixtythree, 3)))
        self.live_time_constant.set(str(truncate_float(taup, 3)))
    
    def get_simulation_params(self, Kp, Ki, Kd):
        return SimulationParams(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            cv_min=self.cv_min.get(),
            cv_max=self.cv_max.get(),
            cv_start=self.cv_start.get(),
            pv_min=self.pv_min.get(),
            pv_max=self.pv_max.get(),
            pv_start=self.pv_start.get(),
            pv_final=self.pv_final.get(),
            process_gain=self.compute_process_gain(),
            dead_time=self.dead_time.get(),
            time_constant=self.compute_time_constant(),
            setpoint=self.setpoint.get(),
            timestep=(1 / (1000 / self.pid_update.get())),
        )

    def run_manual(self):
        starttime = time.time()
        Kp = self.kp_var.get()
        Ki = self.ki_var.get()
        Kd = self.kd_var.get()
        params = self.get_simulation_params(Kp, Ki, Kd)
        sp = self.setpoint.get()
        pv, sp, cv, t_eval = simulate_system(params)
        title = "PID Simulation"
        self.update_plot(t_eval, pv, sp, cv, title)
        sp_val = sp[-1]
        pv_range = self.pv_max.get() - self.pv_min.get()
        max_overshoot_val = max(0, (np.max(pv) - sp_val))
        max_overshoot_pct = (max_overshoot_val / pv_range) * 100
        settling_time_val = compute_settling_time(pv, t_eval, sp_val)
        self.max_overshoot.set(f"{max_overshoot_val:.2f}")
        self.max_overshoot_pct.set(f"{max_overshoot_pct:.2f}%")
        self.settling_time.set(f"{settling_time_val:.2f}s")
        endtime = time.time()
        elapsed_time_val = endtime - starttime
        self.elapsed_time.set(f"{elapsed_time_val:.2f}s")

    def run_optimization(self):
        # Create live plot window
        starttime = time.time()
        def live_cost_function(pid_gains):
            global trial_counter
            Kp, Ki, Kd = pid_gains
            
            params = self.get_simulation_params(Kp, Ki, Kd)
            #t_eval = np.linspace(0, sim_time, num_points)
            #timestep = t_eval[1] - t_eval[0]
            
            # Run the system Simulation
            sp = params.pv_final
            pv, sp, cv, t_eval= simulate_system(params)
            
            # If PV is not a number then reject this solution
            if np.any(np.isnan(pv)) or np.any(np.isinf(pv)):
                return 1e6
            
            # Live plotting
            trial_counter += 1

            # Calculate Error
            error = sp - pv
            
            # Weighting Calculations
            # Integrated Squared Error Calc
            # This is mainly used to avoid scenarios where the pv never reaches the setpoint
            ise = np.sum(np.square(error)) * params.timestep
            
            # Overshoot Penalty Penalty calculation
            overshoot = np.maximum(0, pv - sp)
            overshoot_penalty = np.sum(np.square(overshoot)) * params.timestep
            
            # Large initial PV change penalty
            # Penalize large PV step changes
            initial_window = int(1 / params.timestep)
            initial_pv = pv[0]
            transient_deviation = np.abs(pv[:initial_window] - initial_pv)
            initial_instability_penalty = np.sum(transient_deviation)
            
            # Large initial control actions penalty
            # Penalize large CV step changes
            initial_cv_penalty = np.sum(np.abs(np.diff(cv[:initial_window])))
            
            # Settling time Penalty
            settling_time = compute_settling_time(pv, t_eval, sp[-1])
            
            # Weighting Setpoints - adjust these based on your process to get the best response
            # ISE Weight
            alpha = 1.0
            # Overshoot Penalty Weight
            beta = 10.0
            # Large Initial PV Change Weight
            gamma = 1.0
            # Initial CV Jump Weight
            delta = 1.0
            # Settling Time Weight
            epsilon = 10.0
            
            # Total Cost Calc
            title = (f"Trial #{trial_counter+1} | Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
            cost = (alpha * ise) + (beta * overshoot_penalty) + (gamma * initial_instability_penalty) + (delta * initial_cv_penalty) + (epsilon * settling_time) 
            self.update_plot(t_eval, pv, sp, cv, title)
            return cost

        bounds = [(0, 2), (0, 2), (0, 1)]
        result = differential_evolution(live_cost_function, bounds, seed=42, strategy='best1bin', maxiter=100, popsize=15, tol=1e-6)

        # Final result
        kp, ki, kd = result.x
        params = self.get_simulation_params(kp, ki, kd)
        pv, sp, cv, t_eval= simulate_system(params)
        title = (f"Final PID Optimization | Kp:{kp:.3f}, Ki:{ki:.3f}, Kd:{kd:.3f}, Trials:{trial_counter}")
        self.update_plot(t_eval, pv, sp, cv, title)
        self.kp_var.set(round(kp, 3))
        self.ki_var.set(round(ki, 3))
        self.kd_var.set(round(kd, 3)) 
        sp_val = sp[-1]
        self.setpoint.set(round(sp_val, 3))
        pv_range = self.pv_max.get() - self.pv_min.get()
        max_overshoot_val = max(0, (np.max(pv) - sp_val))
        max_overshoot_pct = (max_overshoot_val / pv_range) * 100
        settling_time_val = compute_settling_time(pv, t_eval, sp_val)
        self.max_overshoot.set(f"{max_overshoot_val:.2f}")
        self.max_overshoot_pct.set(f"{max_overshoot_pct:.2f}%")
        self.settling_time.set(f"{settling_time_val:.2f}s")
        endtime = time.time()
        elapsed_time_val = endtime - starttime
        self.elapsed_time.set(f"{elapsed_time_val:.2f}s")

# Launch GUI
if __name__ == "__main__":
    app = PIDApp()
    app.mainloop()
