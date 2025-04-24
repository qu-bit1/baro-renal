import numpy as np
import matplotlib.pyplot as plt
from renal_model import RenalModel, RenalModelParameters
from renal_tubular import RenalTubular
from neural_mechanisms import NeuralControl
from scipy.integrate import odeint

def main():
    # Initialize model parameters
    params = RenalModelParameters()
    
    # Initialize model components
    renal_model = RenalModel(params)
    tubular_model = RenalTubular(params)
    neural_control = NeuralControl(params)  # Initialize neural control
    renal_model.tubular_model = tubular_model  # Add tubular model to main model
    
    # Set simulation parameters
    t_end = 24 * 60  # 24 hours in minutes
    t = np.linspace(0, t_end, 1000)
    
    # Initial state
    initial_state = {
        # Hemodynamics
        'blood_volume_L': params.blood_volume_nom,
        'cardiac_output_delayed': params.CO_nom,
        'CO_error': 0.0,
        'mean_arterial_pressure': params.nominal_map_setpoint,
        
        # RAAS system
        'renin': params.renin_secretion_rate_nom,
        'angiotensin_I': params.angiotensin_I_nom,
        'angiotensin_II': params.angiotensin_II_nom,
        'aldosterone': params.aldosterone_nom,
        'ACE_activity': params.ACE_activity_nom,
        
        # Vascular parameters
        'preafferent_pressure_autoreg_signal': 1.0,
        'CCB_effect': 1.0,
        'afferent_resistance': params.nom_preafferent_arteriole_resistance,
        'efferent_arteriole_resistance': params.nom_preafferent_arteriole_resistance,
        'peritubular_resistance': params.nom_preafferent_arteriole_resistance,
        
        # Pressures
        'glomerular_pressure': 60.0,
        'Bowmans_capsule_pressure': 15.0,
        
        # Electrolytes and fluids
        'plasma_Na': params.ref_Na_concentration,
        'blood_volume_water': params.blood_volume_nom,
        'plasma_K': 4.0,
        'plasma_osmolarity': 290.0,
        'distal_Na_delivery': 0.0,
        
        # Tubular parameters
        'ADH': 1.0,
        'proximal_tubule_Na_reab_frac': params.prox_tubule_Na_reab_frac_nom,
        'loop_henle_Na_reab_frac': params.loop_henle_Na_reab_frac_nom,
        'distal_tubule_Na_reab_frac': params.distal_tubule_Na_reab_frac_nom,
        'collecting_duct_Na_reab_frac': params.collecting_duct_Na_reab_frac_nom,
        
        # Neural parameters
        'sympathetic_tone': params.sympathetic_tone_nom,
        'parasympathetic_tone': params.parasympathetic_tone_nom,
        'renal_symp_nerve_activity': params.renal_symp_nerve_activity_nom,
        'heart_rate': 72.0,  # Initial heart rate (beats per minute)
        'stroke_volume': 70.0  # Initial stroke volume (ml per beat)
    }
    
    # Convert initial state to list for ODE solver
    state_vector = [
        initial_state['blood_volume_L'],
        initial_state['cardiac_output_delayed'],
        initial_state['CO_error'],
        initial_state['mean_arterial_pressure'],
        initial_state['renin'],
        initial_state['angiotensin_I'],
        initial_state['angiotensin_II'],
        initial_state['aldosterone'],
        initial_state['ACE_activity'],
        initial_state['preafferent_pressure_autoreg_signal'],
        initial_state['CCB_effect'],
        initial_state['afferent_resistance'],
        initial_state['efferent_arteriole_resistance'],
        initial_state['peritubular_resistance'],
        initial_state['glomerular_pressure'],
        initial_state['Bowmans_capsule_pressure'],
        initial_state['plasma_Na'],
        initial_state['blood_volume_water'],
        initial_state['plasma_K'],
        initial_state['plasma_osmolarity'],
        initial_state['distal_Na_delivery'],
        # Add neural parameters to state vector
        initial_state['sympathetic_tone'],
        initial_state['parasympathetic_tone'],
        initial_state['renal_symp_nerve_activity'],
        initial_state['heart_rate'],
        initial_state['stroke_volume']
    ]
    
    # Run simulation
    def derivatives(state, t):
        return renal_model.derivatives(t, state)
    
    solution = odeint(derivatives, state_vector, t)
    
    # Calculate ADH and neural parameters at each time point
    ADH_values = []
    baroreceptor_firing_values = []
    
    for i in range(len(t)):
        # Create state dictionary from solution at time i
        state = {
            'blood_volume_L': solution[i,0],
            'cardiac_output_delayed': solution[i,1],
            'CO_error': solution[i,2],
            'mean_arterial_pressure': solution[i,3],
            'renin': solution[i,4],
            'angiotensin_I': solution[i,5],
            'angiotensin_II': solution[i,6],
            'aldosterone': solution[i,7],
            'ACE_activity': solution[i,8],
            'preafferent_pressure_autoreg_signal': solution[i,9],
            'CCB_effect': solution[i,10],
            'afferent_resistance': solution[i,11],
            'efferent_arteriole_resistance': solution[i,12],
            'peritubular_resistance': solution[i,13],
            'glomerular_pressure': solution[i,14],
            'Bowmans_capsule_pressure': solution[i,15],
            'plasma_Na': solution[i,16],
            'blood_volume_water': solution[i,17],
            'plasma_K': solution[i,18],
            'plasma_osmolarity': solution[i,19],
            'distal_Na_delivery': solution[i,20],
            # Add neural parameters
            'sympathetic_tone': solution[i,21],
            'parasympathetic_tone': solution[i,22],
            'renal_symp_nerve_activity': solution[i,23],
            'heart_rate': solution[i,24],
            'stroke_volume': solution[i,25]
        }
        
        # Add the tubular parameters that are calculated separately
        state.update({
            'ADH': 1.0,  # Initial value, will be updated by tubular model
            'proximal_tubule_Na_reab_frac': params.prox_tubule_Na_reab_frac_nom,
            'loop_henle_Na_reab_frac': params.loop_henle_Na_reab_frac_nom,
            'distal_tubule_Na_reab_frac': params.distal_tubule_Na_reab_frac_nom,
            'collecting_duct_Na_reab_frac': params.collecting_duct_Na_reab_frac_nom
        })
        
        # Calculate neural effects
        neural_effects = neural_control.update_neural_state(state, t[i])
        
        # Calculate renal function with neural effects
        renal = renal_model.calculate_renal_vasculature(state, {
            'mean_arterial_pressure': state['mean_arterial_pressure']
        })
        
        # Calculate tubular function with neural effects
        tubular = tubular_model.calculate_tubular_function(state, renal, t[i], neural_effects['renal_effects'])
        
        # Store values
        ADH_values.append(tubular['ADH'])
        baroreceptor_firing_values.append(neural_effects['autonomic']['baroreceptor_firing_rate'])
    
    # Process results
    results = {
        'time': t,
        'blood_pressure': solution[:, 3],  # mean_arterial_pressure
        'cardiac_output': solution[:, 1],  # cardiac_output_delayed
        'blood_volume': solution[:, 0],  # blood_volume_L
        'plasma_Na': solution[:, 16],  # plasma_Na
        'plasma_K': solution[:, 18],  # plasma_K
        'plasma_osmolarity': solution[:, 19],  # plasma_osmolarity
        'renin': solution[:, 4],  # renin
        'angiotensin_I': solution[:, 5],  # angiotensin_I
        'angiotensin_II': solution[:, 6],  # angiotensin_II
        'aldosterone': solution[:, 7],  # aldosterone
        'ADH': np.array(ADH_values),
        'sympathetic_tone': solution[:, 21],  # sympathetic_tone
        'parasympathetic_tone': solution[:, 22],  # parasympathetic_tone
        'renal_symp_nerve_activity': solution[:, 23],  # renal_symp_nerve_activity
        'heart_rate': solution[:, 24],  # heart_rate
        'stroke_volume': solution[:, 25],  # stroke_volume
        'baroreceptor_firing_rate': np.array(baroreceptor_firing_values)
    }
    
    # Create plots (original 8 panels plus neural mechanisms panels)
    plt.figure(figsize=(15, 25))
    
    # Hemodynamics
    plt.subplot(5, 2, 1)
    plt.plot(t/60, results['blood_pressure'])
    plt.title('Mean Arterial Pressure')
    plt.xlabel('Time (hours)')
    plt.ylabel('Pressure (mmHg)')
    
    plt.subplot(5, 2, 2)
    plt.plot(t/60, results['cardiac_output'])
    plt.title('Cardiac Output')
    plt.xlabel('Time (hours)')
    plt.ylabel('Flow (L/min)')
    
    # Volume and electrolytes
    plt.subplot(5, 2, 3)
    plt.plot(t/60, results['blood_volume'])
    plt.title('Blood Volume')
    plt.xlabel('Time (hours)')
    plt.ylabel('Volume (L)')
    
    plt.subplot(5, 2, 4)
    plt.plot(t/60, results['plasma_Na'])
    plt.title('Plasma Sodium')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mEq/L)')
    
    # RAAS system
    plt.subplot(5, 2, 5)
    plt.plot(t/60, results['renin'], label='Renin')
    plt.plot(t/60, results['angiotensin_I'], label='AngI')
    plt.plot(t/60, results['angiotensin_II'], label='AngII')
    plt.title('Renin-Angiotensin System')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    plt.legend()
    
    plt.subplot(5, 2, 6)
    plt.plot(t/60, results['aldosterone'])
    plt.title('Aldosterone')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    
    # Other hormones and electrolytes
    plt.subplot(5, 2, 7)
    plt.plot(t/60, results['ADH'])
    plt.title('ADH (Vasopressin)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    
    plt.subplot(5, 2, 8)
    plt.plot(t/60, results['plasma_osmolarity'])
    plt.title('Plasma Osmolarity')
    plt.xlabel('Time (hours)')
    plt.ylabel('mOsm/L')
    
    # Neural mechanisms
    plt.subplot(5, 2, 9)
    plt.plot(t/60, results['sympathetic_tone'], label='Sympathetic')
    plt.plot(t/60, results['parasympathetic_tone'], label='Parasympathetic')
    plt.title('Autonomic Tone')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    plt.legend()
    
    plt.subplot(5, 2, 10)
    plt.plot(t/60, results['heart_rate'])
    plt.title('Heart Rate')
    plt.xlabel('Time (hours)')
    plt.ylabel('beats/min')
    
    # Save the expanded figure with neural mechanisms
    plt.tight_layout()
    plt.savefig('renal_simulation_results.png')
    plt.close()
    
    # Create a separate figure for neural mechanisms details
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(t/60, results['sympathetic_tone'], label='Sympathetic')
    plt.plot(t/60, results['parasympathetic_tone'], label='Parasympathetic')
    plt.title('Autonomic Tone')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(t/60, results['heart_rate'], label='Heart Rate')
    plt.plot(t/60, results['stroke_volume'], label='Stroke Volume')
    plt.title('Cardiac Function')
    plt.xlabel('Time (hours)')
    plt.ylabel('HR (bpm) / SV (ml)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(t/60, results['renal_symp_nerve_activity'])
    plt.title('Renal Sympathetic Nerve Activity')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    
    plt.subplot(2, 2, 4)
    plt.plot(t/60, results['baroreceptor_firing_rate'])
    plt.title('Baroreceptor Firing Rate')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Activity')
    
    plt.tight_layout()
    plt.savefig('neural_mechanisms_results.png')
    plt.close()

if __name__ == "__main__":
    main()
