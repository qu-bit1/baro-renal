import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class RenalModelParameters:
    # Systemic parameters
    nominal_map_setpoint: float = 93.0  # mmHg
    CO_nom: float = 5.0  # L/min
    ECF_nom: float = 15.0  # L
    blood_volume_nom: float = 5.0  # L
    Na_intake_rate: float = 100.0/(24.0*60.0)  # mEq/min (100 mEq/day)
    nom_water_intake: float = 2.1  # L/day
    ref_Na_concentration: float = 140.0  # mEq/L
    plasma_protein_concentration: float = 7.0  # g/dl
    P_venous: float = 4.0  # mmHg
    R_venous: float = 3.4  # mmHg
    nom_mean_filling_pressure: float = 7.0  # mmHg
    venous_compliance: float = 0.13

    # Renal parameters
    nom_renal_blood_flow_L_min: float = 1.0  # L/min
    baseline_nephrons: float = 2e6
    nom_Kf: float = 3.9  # nl/min*mmHg
    nom_oncotic_pressure_difference: float = 28.0  # mmHg
    P_renal_vein: float = 4.0  # mmHg
    GFR_nom: float = 120.0  # ml/min
    filtration_fraction_nom: float = 0.2  # Typical 20%

    # Renal vasculature
    nom_preafferent_arteriole_resistance: float = 19.0  # mmHg
    nom_afferent_diameter: float = 1.5e-5  # m
    nom_efferent_diameter: float = 1.1e-5  # m

    # Tubular parameters
    prox_tubule_Na_reab_frac_nom: float = 0.67  # 67% Na reabsorption in proximal tubule
    loop_henle_Na_reab_frac_nom: float = 0.25  # 25% Na reabsorption in loop of Henle
    distal_tubule_Na_reab_frac_nom: float = 0.05  # 5% Na reabsorption in distal tubule
    collecting_duct_Na_reab_frac_nom: float = 0.02  # 2% Na reabsorption in collecting duct
    
    # RAAS parameters
    renin_secretion_rate_nom: float = 1.0  # ng/ml/hr
    ACE_activity_nom: float = 1.0  # Normalized
    angiotensin_I_nom: float = 1.0  # ng/ml
    angiotensin_II_nom: float = 1.0  # ng/ml
    aldosterone_nom: float = 1.0  # ng/dl
    
    # Time constants (minutes)
    tau_renin: float = 60.0  # Renin half-life
    tau_angiotensin_I: float = 1.0  # AngI half-life
    tau_angiotensin_II: float = 2.0  # AngII half-life
    tau_aldosterone: float = 30.0  # Aldosterone half-life
    tau_ADH: float = 30.0  # ADH half-life
    tau_Na_transport: float = 5.0  # Na transport delay
    tau_water_transport: float = 5.0  # Water transport delay

    # Circadian parameters
    circadian_amp: float = 0.1  # 10% amplitude for circadian variations
    circadian_phase: float = 0.0  # Phase shift in radians
    
    # Additional parameters needed for calculations
    tissue_autoreg_scale: float = 1.0
    Kp_CO: float = 1.0
    CO_scale_species: float = 1.0
    Ki_CO: float = 0.1
    AT1_svr_slope: float = 0.5
    nominal_equilibrium_AT1_bound_AngII: float = 1.0
    nom_systemic_arterial_resistance: float = 20.0
    BV_scale_species: float = 1.0
    AT1_preaff_scale: float = 1.0
    AT1_preaff_slope: float = 0.5
    preaff_signal_nonlin_scale: float = 1.0

class RenalModel:
    def __init__(self, params: Optional[RenalModelParameters] = None):
        self.params = params or RenalModelParameters()
        
        # Initialize state variables
        self.state = {
            'blood_volume_L': self.params.blood_volume_nom,
            'cardiac_output_delayed': self.params.CO_nom,
            'CO_error': 0.0,
            'mean_arterial_pressure': self.params.nominal_map_setpoint,
            'renin': self.params.renin_secretion_rate_nom,
            'angiotensin_I': self.params.angiotensin_I_nom,
            'angiotensin_II': self.params.angiotensin_II_nom,
            'aldosterone': self.params.aldosterone_nom,
            'ACE_activity': self.params.ACE_activity_nom,
            'preafferent_pressure_autoreg_signal': 1.0,
            'CCB_effect': 1.0,
            'afferent_resistance': self.params.nom_preafferent_arteriole_resistance,
            'efferent_arteriole_resistance': self.params.nom_preafferent_arteriole_resistance,
            'peritubular_resistance': self.params.nom_preafferent_arteriole_resistance,
            'glomerular_pressure': 60.0,  # Initial glomerular pressure
            'Bowmans_capsule_pressure': 15.0,  # Initial Bowman's capsule pressure
            'plasma_Na': self.params.ref_Na_concentration,
            'blood_volume_water': self.params.blood_volume_nom,
            'plasma_K': 4.0,  # Initial plasma potassium
            'plasma_osmolarity': 290.0,  # Initial plasma osmolarity
            'distal_Na_delivery': 0.0,  # Initial distal Na delivery
            'ADH': 1.0,  # Initial ADH level
            'proximal_tubule_Na_reab_frac': self.params.prox_tubule_Na_reab_frac_nom,
            'loop_henle_Na_reab_frac': self.params.loop_henle_Na_reab_frac_nom,
            'distal_tubule_Na_reab_frac': self.params.distal_tubule_Na_reab_frac_nom,
            'collecting_duct_Na_reab_frac': self.params.collecting_duct_Na_reab_frac_nom
        }
        
    def calculate_systemic_hemodynamics(self, state: Dict[str, float]) -> Dict[str, float]:
        """Calculate systemic hemodynamics including blood pressure and cardiac output"""
        # Calculate tissue autoregulation signal
        tissue_autoregulation_signal = max(0.1, 1 + self.params.tissue_autoreg_scale * 
            ((self.params.Kp_CO/self.params.CO_scale_species) * 
             (state['cardiac_output_delayed'] - self.params.CO_nom) +
             (self.params.Ki_CO/self.params.CO_scale_species) * state['CO_error']))
        
        # Calculate AT1 effect on systemic vascular resistance
        AT1_svr_int = 1 - self.params.AT1_svr_slope * self.params.nominal_equilibrium_AT1_bound_AngII
        AT1_bound_AngII_effect_on_SVR = AT1_svr_int + self.params.AT1_svr_slope * state['angiotensin_II']
        
        # Calculate systemic arterial resistance
        systemic_arterial_resistance = (self.params.nom_systemic_arterial_resistance * 
                                      tissue_autoregulation_signal * 
                                      AT1_bound_AngII_effect_on_SVR)
        
        # Calculate resistance to venous return
        resistance_to_venous_return = ((8 * self.params.R_venous + systemic_arterial_resistance) / 31)
        
        # Calculate mean filling pressure
        mean_filling_pressure = (self.params.nom_mean_filling_pressure + 
                               (state['blood_volume_L']/self.params.BV_scale_species - 
                                self.params.blood_volume_nom)/self.params.venous_compliance)
        
        # Calculate cardiac output
        cardiac_output = mean_filling_pressure / resistance_to_venous_return
        
        # Calculate total peripheral resistance and mean arterial pressure
        total_peripheral_resistance = systemic_arterial_resistance + self.params.R_venous
        mean_arterial_pressure = cardiac_output * total_peripheral_resistance
        
        return {
            'systemic_arterial_resistance': systemic_arterial_resistance,
            'cardiac_output': cardiac_output,
            'mean_arterial_pressure': mean_arterial_pressure,
            'total_peripheral_resistance': total_peripheral_resistance
        }
    
    def calculate_renal_vasculature(self, state: Dict[str, float], 
                                  hemodynamics: Dict[str, float]) -> Dict[str, float]:
        """Calculate renal vasculature parameters including blood flow and resistances"""
        # Calculate AT1 effects on different arterioles
        AT1_preaff_int = 1 - self.params.AT1_preaff_scale/2
        AT1_effect_on_preaff = (AT1_preaff_int + 
                              self.params.AT1_preaff_scale/(1 + np.exp(-(state['angiotensin_II'] - 
                                                                       self.params.nominal_equilibrium_AT1_bound_AngII)/
                                                                      self.params.AT1_preaff_slope)))
        
        # Calculate preafferent arteriole resistance
        preaff_arteriole_signal_multiplier = (AT1_effect_on_preaff * 
                                            state['preafferent_pressure_autoreg_signal'] * 
                                            state['CCB_effect'])
        preaff_arteriole_adjusted_signal_multiplier = (1/(1 + np.exp(self.params.preaff_signal_nonlin_scale * 
                                                                   (1 - preaff_arteriole_signal_multiplier))) + 0.5)
        preafferent_arteriole_resistance = (self.params.nom_preafferent_arteriole_resistance * 
                                          preaff_arteriole_adjusted_signal_multiplier)
        
        # Calculate renal blood flow
        renal_vascular_resistance = (preafferent_arteriole_resistance + 
                                   (state['afferent_resistance'] + 
                                    state['efferent_arteriole_resistance'] + 
                                    state['peritubular_resistance'])/self.params.baseline_nephrons)
        
        renal_blood_flow_L_min = ((hemodynamics['mean_arterial_pressure'] - self.params.P_venous) / 
                                renal_vascular_resistance)
        
        return {
            'renal_blood_flow_L_min': renal_blood_flow_L_min,
            'renal_vascular_resistance': renal_vascular_resistance,
            'preafferent_arteriole_resistance': preafferent_arteriole_resistance
        }
    
    def derivatives(self, t: float, state: List[float]) -> List[float]:
        """Calculate derivatives for the ODE system"""
        # Convert state list to dictionary for easier access
        state_dict = self._state_to_dict(state)
        
        # Calculate hemodynamics
        hemodynamics = self.calculate_systemic_hemodynamics(state_dict)
        
        # Calculate renal vasculature
        renal = self.calculate_renal_vasculature(state_dict, hemodynamics)
        
        # Calculate tubular function
        tubular = self.tubular_model.calculate_tubular_function(state_dict, renal, t)
        
        # Calculate hormonal regulation
        hormones = self.tubular_model.calculate_hormonal_regulation(state_dict, t)
        
        # Calculate derivatives for each state variable
        
        # Blood volume derivative (L/min)
        d_blood_volume = (self.params.nom_water_intake/(24*60) -  # Water intake
                         tubular['final_urine_output']/1000)      # Urine output (convert ml to L)
        
        # Cardiac output delayed derivative (L/min^2)
        d_cardiac_output_delayed = (hemodynamics['cardiac_output'] - 
                                  state_dict['cardiac_output_delayed']) / 60.0  # 1-minute delay
        
        # CO error derivative (L/min^2)
        d_CO_error = hemodynamics['cardiac_output'] - self.params.CO_nom
        
        # Mean arterial pressure derivative (mmHg/min)
        d_mean_arterial_pressure = (hemodynamics['mean_arterial_pressure'] - 
                                  state_dict['mean_arterial_pressure']) / 60.0  # 1-minute delay
        
        # Hormone derivatives with appropriate time constants
        d_renin = (hormones['renin_release'] - state_dict['renin']) / self.params.tau_renin
        d_angiotensin_I = (hormones['angiotensin_I'] - state_dict['angiotensin_I']) / self.params.tau_angiotensin_I
        d_angiotensin_II = (hormones['angiotensin_II'] - state_dict['angiotensin_II']) / self.params.tau_angiotensin_II
        d_aldosterone = (hormones['aldosterone'] - state_dict['aldosterone']) / self.params.tau_aldosterone
        
        # ACE activity derivative (1/min)
        d_ACE_activity = (1.0 - state_dict['ACE_activity']) / 60.0  # Normalize to 1.0 with 1-hour time constant
        
        # Vascular tone derivatives (1/min)
        d_preafferent_pressure_autoreg_signal = (1.0 - state_dict['preafferent_pressure_autoreg_signal']) / 60.0
        d_CCB_effect = (1.0 - state_dict['CCB_effect']) / 60.0
        d_afferent_resistance = (renal['preafferent_arteriole_resistance'] - 
                               state_dict['afferent_resistance']) / 60.0
        d_efferent_resistance = (renal['preafferent_arteriole_resistance'] - 
                               state_dict['efferent_arteriole_resistance']) / 60.0
        d_peritubular_resistance = (renal['preafferent_arteriole_resistance'] - 
                                  state_dict['peritubular_resistance']) / 60.0
        
        # Pressure derivatives (mmHg/min)
        d_glomerular_pressure = (hemodynamics['mean_arterial_pressure'] * 0.6 - 
                               state_dict['glomerular_pressure']) / 60.0
        d_Bowmans_capsule_pressure = (15.0 - state_dict['Bowmans_capsule_pressure']) / 60.0
        
        # Electrolytes and fluids
        d_plasma_Na = (self.params.Na_intake_rate -  # Intake
                      tubular['final_Na_excretion'])  # Excretion
        
        d_blood_volume_water = d_blood_volume  # Water follows blood volume
        
        d_plasma_K = (0.1 -  # Constant K intake
                     0.1 * state_dict['plasma_K']/4.0)  # K excretion proportional to plasma level
        
        d_plasma_osmolarity = (2 * d_plasma_Na +  # Na contributes twice (with accompanying anions)
                              d_plasma_K) / state_dict['blood_volume_L']
        
        # Distal Na delivery derivative (mEq/min)
        d_distal_Na_delivery = (tubular['final_Na_excretion'] - 
                              state_dict['distal_Na_delivery']) / self.params.tau_Na_transport
        
        # Return derivatives in the same order as state variables
        return [
            d_blood_volume,
            d_cardiac_output_delayed,
            d_CO_error,
            d_mean_arterial_pressure,
            d_renin,
            d_angiotensin_I,
            d_angiotensin_II,
            d_aldosterone,
            d_ACE_activity,
            d_preafferent_pressure_autoreg_signal,
            d_CCB_effect,
            d_afferent_resistance,
            d_efferent_resistance,
            d_peritubular_resistance,
            d_glomerular_pressure,
            d_Bowmans_capsule_pressure,
            d_plasma_Na,
            d_blood_volume_water,
            d_plasma_K,
            d_plasma_osmolarity,
            d_distal_Na_delivery
        ]
    
    def _state_to_dict(self, state: List[float]) -> Dict[str, float]:
        """Convert state vector to dictionary"""
        return {
            'blood_volume_L': state[0],
            'cardiac_output_delayed': state[1],
            'CO_error': state[2],
            'mean_arterial_pressure': state[3],
            'renin': state[4],
            'angiotensin_I': state[5],
            'angiotensin_II': state[6],
            'aldosterone': state[7],
            'ACE_activity': state[8],
            'preafferent_pressure_autoreg_signal': state[9],
            'CCB_effect': state[10],
            'afferent_resistance': state[11],
            'efferent_arteriole_resistance': state[12],
            'peritubular_resistance': state[13],
            'glomerular_pressure': state[14],
            'Bowmans_capsule_pressure': state[15],
            'plasma_Na': state[16],
            'blood_volume_water': state[17],
            'plasma_K': state[18],
            'plasma_osmolarity': state[19],
            'distal_Na_delivery': state[20]
        } 