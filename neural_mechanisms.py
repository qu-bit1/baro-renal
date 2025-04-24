import numpy as np
from typing import Dict, Optional

class NeuralControl:
    """
    Neural Control module for renal function and blood pressure regulation model.
    Implements sympathetic and parasympathetic nervous system influences on:
    - Cardiovascular function (heart rate, contractility, vascular tone)
    - Renal function (renal nerve activity, renin release, sodium handling)
    """
    
    def __init__(self, params):
        """Initialize the neural control module with model parameters"""
        self.params = params
        
        # Initialize neural state variables
        self.state = {
            'sympathetic_tone': self.params.sympathetic_tone_nom,
            'parasympathetic_tone': self.params.parasympathetic_tone_nom,
            'renal_symp_nerve_activity': self.params.renal_symp_nerve_activity_nom,
            'baroreceptor_firing_rate': 1.0,
            'heart_rate': 72.0,  # beats per minute
            'stroke_volume': 70.0,  # ml per beat
            'cardiac_contractility': 1.0  # normalized
        }
    
    def calculate_baroreceptor_activity(self, MAP: float) -> float:
        """
        Calculate baroreceptor firing rate based on current mean arterial pressure
        
        Args:
            MAP: Mean arterial pressure in mmHg
            
        Returns:
            Normalized baroreceptor firing rate (0-2, with 1.0 being normal)
        """
        # Baroreceptors increase firing rate with increased pressure (negative feedback)
        # Implement sigmoid-like response between lower and upper thresholds
        
        if MAP >= self.params.baroreceptor_upper_threshold:
            # Maximum inhibition of sympathetic tone at high pressures
            return 2.0
        elif MAP <= self.params.baroreceptor_lower_threshold:
            # Minimum activation at low pressures
            return 0.0
        else:
            # Linear response in the operating range
            normalized_pressure = (MAP - self.params.baroreceptor_lower_threshold) / \
                                (self.params.baroreceptor_upper_threshold - self.params.baroreceptor_lower_threshold)
            # Convert to 0-2 range, with 1.0 at setpoint pressure
            setpoint_fraction = (self.params.nominal_map_setpoint - self.params.baroreceptor_lower_threshold) / \
                               (self.params.baroreceptor_upper_threshold - self.params.baroreceptor_lower_threshold)
            
            # Calculate baroreceptor firing rate with sigmoidal behavior
            # Higher pressure = higher firing rate = more sympathetic inhibition
            return 2.0 / (1.0 + np.exp(-8.0 * (normalized_pressure - setpoint_fraction)))
    
    def calculate_autonomic_tone(self, MAP: float, 
                               current_symp: float, 
                               current_parasymp: float) -> Dict[str, float]:
        """
        Calculate sympathetic and parasympathetic tone based on baroreceptor input
        
        Args:
            MAP: Mean arterial pressure in mmHg
            current_symp: Current sympathetic tone
            current_parasymp: Current parasympathetic tone
            
        Returns:
            Dictionary with updated sympathetic and parasympathetic tone values
        """
        # Calculate baroreceptor firing rate
        baroreceptor_firing = self.calculate_baroreceptor_activity(MAP)
        
        # Baroreceptors inhibit sympathetic and excite parasympathetic tone
        # Apply inverse relationship for sympathetic (higher baroreceptor = lower sympathetic)
        target_symp_tone = self.params.sympathetic_tone_nom * (2.0 - baroreceptor_firing)
        # Apply direct relationship for parasympathetic
        target_parasymp_tone = self.params.parasympathetic_tone_nom * baroreceptor_firing
        
        # Rate of change based on time constants
        d_sympathetic = (target_symp_tone - current_symp) / self.params.tau_symp_response
        d_parasympathetic = (target_parasymp_tone - current_parasymp) / self.params.tau_parasymp_response
        
        # Calculate new values with limits
        new_sympathetic = max(0.1, min(5.0, current_symp + d_sympathetic))
        new_parasympathetic = max(0.1, min(5.0, current_parasymp + d_parasympathetic))
        
        return {
            'sympathetic_tone': new_sympathetic,
            'parasympathetic_tone': new_parasympathetic,
            'baroreceptor_firing_rate': baroreceptor_firing
        }
    
    def calculate_renal_sympathetic_effects(self, symp_tone: float) -> Dict[str, float]:
        """
        Calculate renal-specific sympathetic nervous system effects
        
        Args:
            symp_tone: Current sympathetic tone
            
        Returns:
            Dictionary with renal sympathetic effects
        """
        # Calculate renal sympathetic nerve activity (RSNA)
        # RSNA is proportional to systemic sympathetic tone
        renal_symp_nerve_activity = self.params.renal_symp_nerve_activity_nom * symp_tone
        
        # Calculate effect on renal vascular resistance (vasoconstriction)
        # Higher RSNA = higher resistance
        renal_vasoconstriction = 1.0 + (renal_symp_nerve_activity - 1.0) * self.params.symp_vasoconstriction_gain
        
        # Calculate effect on renin release
        # Higher RSNA = higher renin release
        renin_stimulation = 1.0 + (renal_symp_nerve_activity - 1.0) * self.params.symp_renin_gain
        
        # Calculate effect on tubular sodium reabsorption
        # Higher RSNA = enhanced Na reabsorption
        Na_reabsorption_effect = 1.0 + (renal_symp_nerve_activity - 1.0) * self.params.symp_Na_reabsorption_gain
        
        return {
            'renal_symp_nerve_activity': renal_symp_nerve_activity,
            'renal_vasoconstriction': renal_vasoconstriction,
            'renin_stimulation': renin_stimulation,
            'Na_reabsorption_effect': Na_reabsorption_effect
        }
    
    def calculate_cardiac_effects(self, symp_tone: float, parasymp_tone: float) -> Dict[str, float]:
        """
        Calculate cardiac effects of autonomic nervous system
        
        Args:
            symp_tone: Current sympathetic tone
            parasymp_tone: Current parasympathetic tone
            
        Returns:
            Dictionary with cardiac effects including heart rate and contractility
        """
        # Calculate heart rate (HR)
        # Sympathetic increases HR, parasympathetic decreases HR
        baseline_HR = 72.0  # beats per minute
        HR_symp_effect = (symp_tone - 1.0) * self.params.symp_HR_gain * baseline_HR
        HR_parasymp_effect = (parasymp_tone - 1.0) * self.params.parasymp_HR_gain * baseline_HR
        heart_rate = baseline_HR + HR_symp_effect + HR_parasymp_effect
        
        # Ensure heart rate stays within physiological limits
        heart_rate = max(40.0, min(180.0, heart_rate))  # beats per minute
        
        # Calculate cardiac contractility (affected mainly by sympathetic tone)
        # Higher sympathetic tone = higher contractility
        contractility = 1.0 + (symp_tone - 1.0) * self.params.symp_contractility_gain
        
        # Calculate stroke volume (affected by contractility)
        # Baseline stroke volume modulated by contractility
        baseline_SV = 70.0  # ml per beat
        stroke_volume = baseline_SV * contractility
        
        return {
            'heart_rate': heart_rate,
            'contractility': contractility,
            'stroke_volume': stroke_volume,
            'cardiac_effect': heart_rate * stroke_volume / 5000.0  # Normalized to ~1.0
        }
    
    def calculate_vascular_effects(self, symp_tone: float) -> Dict[str, float]:
        """
        Calculate vascular effects of sympathetic nervous system
        
        Args:
            symp_tone: Current sympathetic tone
            
        Returns:
            Dictionary with vascular effects
        """
        # Calculate systemic vascular resistance effect
        # Higher sympathetic tone = higher resistance (vasoconstriction)
        svr_effect = 1.0 + (symp_tone - 1.0) * self.params.symp_vasoconstriction_gain
        
        # Calculate venous tone effect (affects venous return)
        # Higher sympathetic tone = higher venous tone = increased venous return
        venous_tone = 1.0 + (symp_tone - 1.0) * 0.3
        
        return {
            'svr_effect': svr_effect,
            'venous_tone': venous_tone
        }
    
    def update_neural_state(self, state: Dict[str, float], t: float) -> Dict[str, float]:
        """
        Update neural state based on current physiological state
        
        Args:
            state: Current state dictionary from the main model
            t: Current time in minutes
            
        Returns:
            Dictionary with updated neural effects to be applied to the model
        """
        # Extract relevant state variables
        MAP = state.get('mean_arterial_pressure', self.params.nominal_map_setpoint)
        current_symp = state.get('sympathetic_tone', self.params.sympathetic_tone_nom)
        current_parasymp = state.get('parasympathetic_tone', self.params.parasympathetic_tone_nom)
        
        # Calculate autonomic tone from baroreceptor input
        autonomic = self.calculate_autonomic_tone(MAP, current_symp, current_parasymp)
        
        # Calculate renal sympathetic effects
        renal_effects = self.calculate_renal_sympathetic_effects(autonomic['sympathetic_tone'])
        
        # Calculate cardiac effects
        cardiac = self.calculate_cardiac_effects(
            autonomic['sympathetic_tone'], 
            autonomic['parasympathetic_tone'])
        
        # Calculate vascular effects
        vascular = self.calculate_vascular_effects(autonomic['sympathetic_tone'])
        
        # Update internal state
        self.state.update({
            'sympathetic_tone': autonomic['sympathetic_tone'],
            'parasympathetic_tone': autonomic['parasympathetic_tone'],
            'renal_symp_nerve_activity': renal_effects['renal_symp_nerve_activity'],
            'baroreceptor_firing_rate': autonomic['baroreceptor_firing_rate'],
            'heart_rate': cardiac['heart_rate'],
            'stroke_volume': cardiac['stroke_volume'],
            'cardiac_contractility': cardiac['contractility']
        })
        
        # Combine all effects into one dictionary to return
        return {
            'autonomic': autonomic,
            'renal_effects': renal_effects,
            'cardiac': cardiac,
            'vascular': vascular
        }