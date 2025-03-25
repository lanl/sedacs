
from torch import Tensor
import torch
from dataclasses import dataclass
from ase import units
from typing import Any
import numpy as np

__all__ = ["SystemState", "NVEState", "create_NVE_integrator", "create_NVT_integrator"]

@torch.compile
def map2central(coordinates, cell, inv_cell):
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    coordinates_cell = torch.matmul(coordinates, inv_cell)
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor()
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return torch.matmul(coordinates_cell, cell)

def maxwell_boltzman_dist(masses, temp_in_K):
    temp = units.kB * temp_in_K
    xi = torch.randn((len(masses), 3), device=masses.device)
    momenta = xi * torch.sqrt(masses * temp)[:, None]
    return momenta / masses[:, None]

class Serializable:
    def save(self, file_path):
        """
        Save the instance to a file using torch.save.
        The method saves all attributes in the __dict__ of the instance.
        """
        # Extract all attributes from the instance's __dict__
        data = {key: value for key, value in self.__dict__.items()}
        
        # Use torch.save to serialize the dictionary
        torch.save(data, file_path)

    @classmethod
    def load(cls, file_path, device="cpu", dtype=torch.float32):
        """
        Load an instance from a file using torch.load.
        This method recreates the instance by loading the saved attributes.
        """
        # Load the saved dictionary from file
        data = torch.load(file_path, map_location=torch.device('cpu'))

        # Create an instance of the class
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        
        # Set the loaded attributes to the new instance
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.dtype in [torch.float32, torch.float64]:
                    value = value.to(dtype)
                value = value.to(device)
            setattr(instance, key, value)
        
        return instance
    
class SystemState(Serializable):
    def __init__(self, positions, types, masses, cell, use_shadow=False):
        self.positions = positions
        self.masses = masses
        self.cell = cell
        self.cell_lengths = torch.norm(cell, dim=1)
        self.inv_cell = torch.inverse(cell)
        self.types = types
        self.nats = len(masses)
        self.velocities = torch.zeros_like(positions)
        self.forces = torch.zeros_like(positions)
        self.energy = 0.0
        self.charges = torch.zeros_like(masses)
        self.use_shadow = use_shadow
        if self.use_shadow:
            self.coefficients = torch.tensor((-14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0)).to(positions.dtype).to(positions.device)
            self.p_history = torch.zeros((7, len(positions)), dtype=positions.dtype, device=positions.device)
            self.kappa = 1.84
            self.alpha = 0.0055
            self.delta_p = torch.zeros_like(masses)

    def set_positions(self, positions):
        self.positions = map2central(positions, self.cell, self.inv_cell)

    def get_potential_energy(self):
        return self.energy

    def get_kinetic_energy(self):
        return float((0.5 * torch.sum(self.masses * torch.sum(self.velocities**2, axis=1))).item())
    
    def get_total_energy(self):
        return self.get_potential_energy() + self.get_kinetic_energy()
    
    def get_temperature(self):
        return self.get_kinetic_energy() / (1.5 * units.kB) / self.nats
        



@dataclass
class NVEState:
    system: SystemState
    extra: Any = None

def create_NVE_integrator(calculate_energy_and_forces, dt, use_shadow=False):
    half_dt = dt * 0.5

    def init(system, temp_in_K=0.0):
        if temp_in_K > 0:
            system.velocities = maxwell_boltzman_dist(system.masses, temp_in_K)
        else:
            system.velocities = system.velocities * 0.0
        calculate_energy_and_forces(system, init=True)
        return NVEState(system)
    
    def step(state):
        system = state.system
        new_vel = system.velocities + half_dt * system.forces / system.masses[:, None]
        new_pos = system.positions + new_vel * dt
        system.set_positions(new_pos)
        calculate_energy_and_forces(system, init=False)
        new_vel = new_vel + half_dt * system.forces / system.masses[:, None]
        system.velocities = new_vel
        return state
    return init, step

@dataclass
class NVTState:
    system: SystemState
    extra: Any = None

def create_NVT_integrator(calculate_energy_and_forces, dt, friction, target_temp_in_K, fix_cm=True):
    target_temp = target_temp_in_K * units.kB
    

    def init(system, temp_in_K=0.0):
        if temp_in_K > 0:
            system.velocities = maxwell_boltzman_dist(system.masses, temp_in_K)
        else:
            system.velocities = system.velocities * 0.0
        calculate_energy_and_forces(system, init=True)
        extra = dict()
        extra['sigma'] = torch.sqrt(2 * target_temp * friction / system.masses)[:, None]
        extra['c1'] = dt / 2. - dt * dt * friction / 8.
        extra['c2'] = dt * friction / 2 - dt * dt * friction * friction / 8.
        extra['c3'] = np.sqrt(dt) * extra['sigma'] / 2. - dt**1.5 * friction * extra['sigma'] / 8.

        extra['c5'] = dt**1.5 * extra['sigma'] / (2 * np.sqrt(3))
        extra['c4'] = friction / 2. * extra['c5']
        return NVTState(system, extra)
    
    def step(state):
        system = state.system
        extra = state.extra
        xi = torch.randn_like(system.velocities)
        eta = torch.randn_like(system.velocities)
        rnd_pos = extra['c5'] * eta
        rnd_vel = extra['c3'] * xi - extra['c4'] * eta

        if fix_cm:
            rnd_pos -= rnd_pos.sum(dim=0)[None,:] / system.nats
            rnd_vel -= (rnd_vel *
                             system.masses[:, None]).sum(dim=0)[None,:] / (system.masses[:, None] * system.nats) 
        # First halfstep in the velocity.
        new_vel = system.velocities + (extra['c1'] * system.forces / system.masses[:, None] - extra['c2'] * system.velocities +
                   rnd_vel)  
        new_pos = system.positions + new_vel * dt + rnd_pos
        system.set_positions(new_pos)
        calculate_energy_and_forces(system, init=False) 

        new_vel = new_vel + (extra['c1'] * system.forces / system.masses[:, None] - extra['c2'] * new_vel +
                   rnd_vel) 
        system.velocities = new_vel       
        
        return state

    return init, step