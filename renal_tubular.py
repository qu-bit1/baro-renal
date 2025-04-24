import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class RenalTubular:
    def __init__(self, params):
        self.params = params
        
    def calculate_tubular_function(self, state: Dict[str, float], 
                                 renal_flow: Dict[str, float],
                                 t: float,
                                 neural_effects: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate tubular function including sodium and water handling with neural influences"""
        # Apply circadian variation to GFR
        circadian_factor = self._calculate_circadian_factor(t)
        
        # Get sympathetic tone for ADH calculation if available
        symp_tone = 1.0
        if neural_effects and 'renal_symp_nerve_activity' in neural_effects:
            symp_tone = neural_effects['renal_symp_nerve_activity']
        
        # Calculate ADH level based on current state
        ADH = self._calculate_ADH(state['plasma_osmolarity'], 
                                state['mean_arterial_pressure'],
                                symp_tone)
        
        # Calculate glomerular filtration rate (GFR) with circadian variation
        GFR = (self.params.nom_Kf * 
               (state['glomerular_pressure'] - 
                state['Bowmans_capsule_pressure'] - 
                self.params.nom_oncotic_pressure_difference) * circadian_factor)
        
        # Calculate filtered load
        filtered_Na = GFR * state['plasma_Na']  # mEq/min
        filtered_water = GFR  # ml/min
        
        # Get neural effects on sodium reabsorption if provided
        Na_reabsorption_effect = 1.0
        if neural_effects:
            Na_reabsorption_effect = neural_effects.get('Na_reabsorption_effect', 1.0)
        
        # Calculate proximal tubule reabsorption with hormonal and neural modulation
        proximal_tubule = self._calculate_proximal_tubule_reabsorption(
            filtered_Na, filtered_water, state['angiotensin_II'], Na_reabsorption_effect)
        
        # Calculate loop of Henle function with hormonal modulation
        loop_of_henle = self._calculate_loop_of_henle(
            proximal_tubule['Na_out'],
            proximal_tubule['water_out'],
            ADH)  # Use calculated ADH
        
        # Calculate distal tubule and collecting duct function with neural influence
        distal_function = self._calculate_distal_function(
            loop_of_henle['Na_out'],
            loop_of_henle['water_out'],
            state['aldosterone'],
            ADH,
            Na_reabsorption_effect)  # Include neural effect
        
        return {
            'GFR': GFR,
            'filtered_Na': filtered_Na,
            'filtered_water': filtered_water,
            'proximal_reabsorption': proximal_tubule,
            'loop_of_henle': loop_of_henle,
            'distal_function': distal_function,
            'final_urine_output': distal_function['urine_flow'],
            'final_Na_excretion': distal_function['urine_Na'],
            'ADH': ADH  # Include calculated ADH in return value
        }
    
    def _calculate_proximal_tubule_reabsorption(self, Na_in: float, 
                                              water_in: float,
                                              angiotensin_II: float,
                                              neural_effect: float = 1.0) -> Dict[str, float]:
        """Calculate proximal tubule sodium and water reabsorption with AngII and neural modulation"""
        # AngII enhances Na reabsorption in proximal tubule
        angII_effect = 1.0 + 0.3 * (angiotensin_II - self.params.angiotensin_II_nom)
        
        # Calculate Na reabsorption with neural influence (sympathetic activity enhances Na reabsorption)
        base_Na_reab = self.params.prox_tubule_Na_reab_frac_nom * Na_in
        Na_reabsorption = base_Na_reab * angII_effect * neural_effect
        Na_out = Na_in - Na_reabsorption
        
        # Water follows sodium (glomerulotubular balance)
        water_reabsorption = water_in * (Na_reabsorption / Na_in)
        water_out = water_in - water_reabsorption
        
        return {
            'Na_out': Na_out,
            'water_out': water_out,
            'Na_reabsorption': Na_reabsorption,
            'water_reabsorption': water_reabsorption
        }
    
    def _calculate_loop_of_henle(self, Na_in: float, 
                               water_in: float,
                               ADH: float) -> Dict[str, float]:
        """Calculate loop of Henle function including countercurrent multiplication"""
        # Calculate Na reabsorption in thick ascending limb
        base_Na_reab = self.params.loop_henle_Na_reab_frac_nom * Na_in
        Na_reabsorption = base_Na_reab
        Na_out = Na_in - Na_reabsorption
        
        # Water reabsorption in descending limb (enhanced by ADH)
        ADH_effect = 1.0 + 0.5 * (ADH - 1.0)
        water_reabsorption = water_in * self.params.loop_henle_Na_reab_frac_nom * ADH_effect
        water_out = water_in - water_reabsorption
        
        # Calculate medullary concentration gradient
        medullary_gradient = Na_reabsorption / water_out  # mEq/ml
        
        return {
            'Na_out': Na_out,
            'water_out': water_out,
            'Na_reabsorption': Na_reabsorption,
            'water_reabsorption': water_reabsorption,
            'medullary_gradient': medullary_gradient
        }
    
    def _calculate_distal_function(self, Na_in: float, 
                                 water_in: float,
                                 aldosterone: float,
                                 ADH: float,
                                 neural_effect: float = 1.0) -> Dict[str, float]:
        """Calculate distal tubule and collecting duct function with hormone and neural effects"""
        # Aldosterone effect on Na reabsorption
        aldosterone_effect = 1.0 + 0.5 * (aldosterone - self.params.aldosterone_nom)
        
        # ADH effect on water reabsorption
        ADH_effect = 1.0 + 0.8 * (ADH - 1.0)
        
        # Distal tubule Na reabsorption with neural influence
        # Sympathetic activity enhances distal sodium reabsorption
        distal_Na_reab = (self.params.distal_tubule_Na_reab_frac_nom * 
                         Na_in * aldosterone_effect * neural_effect)
        
        # Collecting duct function with neural influence
        cd_Na_reab = (self.params.collecting_duct_Na_reab_frac_nom * 
                     (Na_in - distal_Na_reab) * aldosterone_effect * neural_effect)
        
        # Final Na excretion
        final_Na = Na_in - distal_Na_reab - cd_Na_reab
        
        # Water reabsorption follows ADH
        water_reabsorption = water_in * (1.0 - 0.1) * ADH_effect
        final_water = water_in - water_reabsorption
        
        return {
            'Na_out': final_Na,
            'water_out': final_water,
            'urine_flow': final_water,
            'urine_Na': final_Na,
            'Na_reabsorption': distal_Na_reab + cd_Na_reab,
            'water_reabsorption': water_reabsorption
        }
    
    def calculate_hormonal_regulation(self, state: Dict[str, float], 
                                    t: float,
                                    neural_effects: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate hormonal regulation including RAAS and ADH with neural influences"""
        # Apply circadian variation
        circadian_factor = self._calculate_circadian_factor(t)
        
        # Calculate distal Na delivery if not provided
        distal_Na_delivery = state.get('distal_Na_delivery', 
            state['plasma_Na'] * self.params.GFR_nom * 
            (1 - self.params.prox_tubule_Na_reab_frac_nom))  # Estimate based on GFR and proximal reabsorption
        
        # Get ACE activity or use default
        ACE_activity = state.get('ACE_activity', 1.0)  # Default to normal ACE activity
        
        # Get renal sympathetic neural effects (if provided)
        renin_stimulation = 1.0
        if neural_effects:
            renin_stimulation = neural_effects.get('renin_stimulation', 1.0)
        
        # Calculate renin release based on renal perfusion pressure, salt delivery,
        # and sympathetic neural activity
        renin_release = self._calculate_renin_release(
            state['mean_arterial_pressure'],
            distal_Na_delivery,
            circadian_factor,
            renin_stimulation)
        
        # Calculate angiotensin I from renin
        angiotensin_I = self._calculate_angiotensin_I(renin_release)
        
        # Convert AngI to AngII through ACE
        angiotensin_II = self._calculate_angiotensin_II(
            angiotensin_I, 
            ACE_activity)
        
        # Calculate aldosterone based on AngII and potassium
        aldosterone = self._calculate_aldosterone(
            angiotensin_II, 
            state['plasma_K'],
            circadian_factor)
        
        # Calculate ADH based on osmolarity and blood pressure
        # Include sympathetic influence if available
        symp_tone = 1.0
        if neural_effects:
            symp_tone = neural_effects.get('renal_symp_nerve_activity', 1.0)
        
        ADH = self._calculate_ADH(
            state['plasma_osmolarity'],
            state['mean_arterial_pressure'],
            symp_tone)
        
        return {
            'renin_release': renin_release,
            'angiotensin_I': angiotensin_I,
            'angiotensin_II': angiotensin_II,
            'aldosterone': aldosterone,
            'ADH': ADH,
            'distal_Na_delivery': distal_Na_delivery,  # Include in return value for next iteration
            'ACE_activity': ACE_activity  # Include in return value for next iteration
        }
    
    def _calculate_circadian_factor(self, t: float) -> float:
        """Calculate circadian variation factor"""
        # Convert time to hours and calculate phase
        hours = (t / 60.0) % 24
        phase = 2 * np.pi * hours / 24 + self.params.circadian_phase
        
        # Calculate circadian factor with peak in early morning
        return 1.0 + self.params.circadian_amp * np.sin(phase)
    
    def _calculate_renin_release(self, MAP: float, 
                               distal_Na_delivery: float,
                               circadian_factor: float,
                               neural_effect: float = 1.0) -> float:
        """Calculate renin release with multiple inputs and neural influence"""
        # Pressure effect (inverse relationship)
        pressure_effect = 1.0 + 2.0 * (self.params.nominal_map_setpoint - MAP) / self.params.nominal_map_setpoint
        
        # Macula densa feedback (inverse relationship with Na delivery)
        md_effect = 1.0 + 0.5 * (1.0 - distal_Na_delivery / self.params.GFR_nom)
        
        # Combine effects with circadian variation and neural effects
        # Sympathetic activity stimulates renin release (neural_effect > 1.0)
        renin_release = (self.params.renin_secretion_rate_nom * 
                        pressure_effect * md_effect * circadian_factor * neural_effect)
        
        return max(0.1, min(5.0, renin_release))
    
    def _calculate_angiotensin_I(self, renin: float) -> float:
        """Convert renin to angiotensin I"""
        return renin * self.params.angiotensin_I_nom
    
    def _calculate_angiotensin_II(self, angiotensin_I: float, 
                                 ACE_activity: float) -> float:
        """Convert angiotensin I to angiotensin II based on ACE activity"""
        return angiotensin_I * ACE_activity
    
    def _calculate_aldosterone(self, angiotensin_II: float, 
                             plasma_K: float,
                             circadian_factor: float) -> float:
        """Calculate aldosterone with AngII, potassium, and circadian inputs"""
        # AngII effect
        angII_effect = 1.0 + 0.5 * (angiotensin_II - self.params.angiotensin_II_nom)
        
        # Potassium effect
        K_effect = 1.0 + 0.3 * (plasma_K - 4.0)  # Normal K = 4.0 mEq/L
        
        # Combine effects with circadian variation
        aldosterone = (self.params.aldosterone_nom * 
                      angII_effect * K_effect * circadian_factor)
        
        return max(0.1, min(5.0, aldosterone))
    
    def _calculate_ADH(self, plasma_osmolarity: float, 
                     MAP: float,
                     sympathetic_tone: float = 1.0) -> float:
        """Calculate ADH (vasopressin) release with neural input"""
        # Osmolarity effect (positive relationship)
        osm_effect = 1.0 + 0.5 * (plasma_osmolarity - 290) / 290  # Normal osmolarity = 290 mOsm/L
        
        # Pressure effect (inverse relationship)
        pressure_effect = 1.0 - 0.3 * (MAP - self.params.nominal_map_setpoint) / self.params.nominal_map_setpoint
        
        # Sympathetic effect (high sympathetic tone can increase ADH, especially during stress)
        symp_effect = 1.0 + 0.2 * (sympathetic_tone - 1.0)
        
        ADH = self.params.aldosterone_nom * osm_effect * pressure_effect * symp_effect
        return max(0.1, min(5.0, ADH))