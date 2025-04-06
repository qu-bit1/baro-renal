import numpy as np
import matplotlib.pyplot as plt
from renal_model import RenalModel, RenalModelParameters
from renal_tubular import RenalTubular
from scipy.integrate import odeint

def main():
    # Initialize model parameters
    params = RenalModelParameters()
    
    # Initialize model components
    renal_model = RenalModel(params)
    tubular_model = RenalTubular(params)
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
        'collecting_duct_Na_reab_frac': params.collecting_duct_Na_reab_frac_nom
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
        initial_state['distal_Na_delivery']
    ]
    
    # Run simulation
    def derivatives(state, t):
        return renal_model.derivatives(t, state)
    
    solution = odeint(derivatives, state_vector, t)
    
    # Calculate ADH at each time point
    ADH_values = []
    for i in range(len(t)):
        # Only include the ODE state variables in the dictionary
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
            'distal_Na_delivery': solution[i,20]
        }
        
        # Add the tubular parameters that are calculated separately
        state.update({
            'ADH': 1.0,  # Initial value, will be updated by tubular model
            'proximal_tubule_Na_reab_frac': params.prox_tubule_Na_reab_frac_nom,
            'loop_henle_Na_reab_frac': params.loop_henle_Na_reab_frac_nom,
            'distal_tubule_Na_reab_frac': params.distal_tubule_Na_reab_frac_nom,
            'collecting_duct_Na_reab_frac': params.collecting_duct_Na_reab_frac_nom
        })
        
        renal = renal_model.calculate_renal_vasculature(state, {
            'mean_arterial_pressure': state['mean_arterial_pressure']
        })
        tubular = tubular_model.calculate_tubular_function(state, renal, t[i])
        ADH_values.append(tubular['ADH'])
    
    # Process results
    results = {
        'time': t,
        'blood_pressure': solution[:, list(initial_state.keys()).index('mean_arterial_pressure')],
        'cardiac_output': solution[:, list(initial_state.keys()).index('cardiac_output_delayed')],
        'blood_volume': solution[:, list(initial_state.keys()).index('blood_volume_L')],
        'plasma_Na': solution[:, list(initial_state.keys()).index('plasma_Na')],
        'plasma_K': solution[:, list(initial_state.keys()).index('plasma_K')],
        'plasma_osmolarity': solution[:, list(initial_state.keys()).index('plasma_osmolarity')],
        'renin': solution[:, list(initial_state.keys()).index('renin')],
        'angiotensin_I': solution[:, list(initial_state.keys()).index('angiotensin_I')],
        'angiotensin_II': solution[:, list(initial_state.keys()).index('angiotensin_II')],
        'aldosterone': solution[:, list(initial_state.keys()).index('aldosterone')],
        'ADH': np.array(ADH_values)
    }
    
    # Create plots
    plt.figure(figsize=(15, 20))
    
    # Hemodynamics
    plt.subplot(4, 2, 1)
    plt.plot(t/60, results['blood_pressure'])
    plt.title('Mean Arterial Pressure')
    plt.xlabel('Time (hours)')
    plt.ylabel('Pressure (mmHg)')
    
    plt.subplot(4, 2, 2)
    plt.plot(t/60, results['cardiac_output'])
    plt.title('Cardiac Output')
    plt.xlabel('Time (hours)')
    plt.ylabel('Flow (L/min)')
    
    # Volume and electrolytes
    plt.subplot(4, 2, 3)
    plt.plot(t/60, results['blood_volume'])
    plt.title('Blood Volume')
    plt.xlabel('Time (hours)')
    plt.ylabel('Volume (L)')
    
    plt.subplot(4, 2, 4)
    plt.plot(t/60, results['plasma_Na'])
    plt.title('Plasma Sodium')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mEq/L)')
    
    # RAAS system
    plt.subplot(4, 2, 5)
    plt.plot(t/60, results['renin'], label='Renin')
    plt.plot(t/60, results['angiotensin_I'], label='AngI')
    plt.plot(t/60, results['angiotensin_II'], label='AngII')
    plt.title('Renin-Angiotensin System')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    plt.legend()
    
    plt.subplot(4, 2, 6)
    plt.plot(t/60, results['aldosterone'])
    plt.title('Aldosterone')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    
    # Other hormones and electrolytes
    plt.subplot(4, 2, 7)
    plt.plot(t/60, results['ADH'])
    plt.title('ADH (Vasopressin)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Normalized Level')
    
    plt.subplot(4, 2, 8)
    plt.plot(t/60, results['plasma_osmolarity'])
    plt.title('Plasma Osmolarity')
    plt.xlabel('Time (hours)')
    plt.ylabel('mOsm/L')
    
    plt.tight_layout()
    plt.savefig('renal_simulation_results.png')
    plt.close()

if __name__ == "__main__":
    main() 