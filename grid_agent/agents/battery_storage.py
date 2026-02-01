"""
Battery Storage Agent
Manages battery energy storage systems for grid support
"""

import pandapower as pp
from typing import Dict, List, Any, Optional


class BatteryStorageAgent:
    """
    Battery Energy Storage System (BESS) Agent
    Provides grid support through energy storage
    """
    
    def __init__(self):
        self.name = "battery_storage"
        self.batteries = {}  # Track added batteries
    
    def add_battery(self, net, bus_id: int, capacity_mw: float = 1.0, 
                    soc: float = 0.5, name: str = None) -> Dict:
        """
        Add a battery storage unit at a bus
        
        Args:
            net: pandapower network
            bus_id: Bus to connect battery
            capacity_mw: Battery capacity in MW
            soc: State of charge (0-1)
            name: Optional battery name
        """
        try:
            if bus_id not in net.bus.index:
                return {"success": False, "error": f"Bus {bus_id} not found"}
            
            # Create storage element
            storage_idx = pp.create_storage(
                net,
                bus=bus_id,
                p_mw=0,  # Initial power (positive = discharging)
                max_e_mwh=capacity_mw * 4,  # 4 hour duration
                soc_percent=soc * 100,
                name=name or f"Battery_Bus_{bus_id}"
            )
            
            battery_info = {
                "idx": storage_idx,
                "bus": bus_id,
                "capacity_mw": capacity_mw,
                "soc": soc,
                "status": "idle"
            }
            self.batteries[storage_idx] = battery_info
            
            return {"success": True, "battery_id": storage_idx, "info": battery_info}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def discharge_battery(self, net, battery_id: int, power_mw: float) -> Dict:
        """Discharge battery (inject power to grid)"""
        try:
            if battery_id not in net.storage.index:
                return {"success": False, "error": "Battery not found"}
            
            net.storage.at[battery_id, "p_mw"] = power_mw
            self.batteries[battery_id]["status"] = "discharging"
            
            return {"success": True, "power_mw": power_mw}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def charge_battery(self, net, battery_id: int, power_mw: float) -> Dict:
        """Charge battery (absorb power from grid)"""
        try:
            if battery_id not in net.storage.index:
                return {"success": False, "error": "Battery not found"}
            
            net.storage.at[battery_id, "p_mw"] = -power_mw  # Negative = charging
            self.batteries[battery_id]["status"] = "charging"
            
            return {"success": True, "power_mw": -power_mw}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_battery_status(self, net) -> List[Dict]:
        """Get status of all batteries in network"""
        status = []
        if hasattr(net, 'storage') and len(net.storage) > 0:
            for idx in net.storage.index:
                status.append({
                    "id": int(idx),
                    "bus": int(net.storage.at[idx, "bus"]),
                    "name": net.storage.at[idx, "name"],
                    "p_mw": float(net.storage.at[idx, "p_mw"]),
                    "soc_percent": float(net.storage.at[idx, "soc_percent"]),
                    "max_e_mwh": float(net.storage.at[idx, "max_e_mwh"])
                })
        return status
    
    def remove_battery(self, net, battery_id: int) -> Dict:
        """Remove a battery from the network"""
        try:
            if battery_id in net.storage.index:
                net.storage.drop(battery_id, inplace=True)
                if battery_id in self.batteries:
                    del self.batteries[battery_id]
                return {"success": True}
            return {"success": False, "error": "Battery not found"}
        except Exception as e:
            return {"success": False, "error": str(e)}
