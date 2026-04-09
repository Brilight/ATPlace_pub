import os
import csv
import time
import json
import yaml
import subprocess

import gdspy
import numpy as np

import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.interpolate import griddata

import utils.fill_space


class Thermal_solver():
    def __init__(self, system, decimal=3, thermal_root_path=None):
        '''
        All the dimensions in the Thermal_solver are mm
        '''
        self.path = thermal_root_path
        self.num_chiplets = system.num_chiplets
        self.num_nodes = system.num_nodes
        self.granularity = system.granularity / 1e3
        self.decimal = decimal
        self.intp_width = np.round((system.intp_width) / 1e3, self.decimal)
        self.intp_height = np.round((system.intp_height) / 1e3, self.decimal)
        self.num_grid_x = system.num_grid_x
        self.num_grid_y = system.num_grid_y
        
    def set_pos(self, power, pos, width, height):
        num_chiplets = self.num_chiplets
        num_nodes = self.num_nodes
        self.power = power
        self.x = np.round((pos[0]-width/2) / 1e3, self.decimal)
        self.y = np.round((pos[1]-height/2) / 1e3, self.decimal)
        self.width = np.round(width / 1e3, self.decimal)
        self.height = np.round(height / 1e3, self.decimal)

    def gen_flp_and_power(self):
        pass
    
    def run(self, filename):
        pass
    
class ATSim_solver(Thermal_solver):
    def __init__(self, system):
        super(ATSim_solver, self).__init__(system, 1, system.thermal_dir)
        
    def read_configs(self, json_file):
        with open(json_file, 'r') as f:
            configs = json.load(f)
        self.layers = configs
        self.max_depth = configs['max_depth']
        
    def gen_stack(self, dir_name):
        """
        Generate the <Component> section of the XML file with Chiplet_<id> entries.
        Each chiplet includes three layers: Bond, Act, and Sub.
        """
        thermal_dir = self.path if self.path else "./"
        z_origin = 0

        def add_layer(component, name, layer_info):
            mat, thick = layer_info[:2]
            layer = ET.SubElement(component, "Layer", attrib={
                "name": name, "FillingMaterial": mat,
            })
            ET.SubElement(layer, "Geometry", attrib={
                "Thickness": str(thick)
            })
            return layer

        # Create the root element for the XML
        package = ET.Element("Package", attrib={
            "name": "SYSTEM",
            "lengthunit": "m"
        })

        # Add Location and SimConfig elements
        location = ET.SubElement(package, "Location", attrib={
            "offsetX": "0",
            "offsetY": "0",
            "orientation": "R0"
        })
        ET.SubElement(package, "MaterialLib", attrib={
            "File": str(os.path.join(thermal_dir, "material.config"))
        })
        ET.SubElement(package, "PowerLib", attrib={
            "File": str(os.path.join(thermal_dir, f"{dir_name}/powerlib.yml"))
        })
        ET.SubElement(package, "TempData", attrib={
            "File": str(os.path.join(thermal_dir, f"{dir_name}/raw/steadytemp.txt"))
        })

        # Interposer component
        component1 = ET.SubElement(package, "Component", attrib={
            "number": "1",
            "type": "Interposer"
        })
        ET.SubElement(component1, "Geometry", attrib={
            "Length": str(self.intp_width/1e3),
            "Width": str(self.intp_height/1e3),
            "XYorigin": "Center"
        })
        
        for layer_name in self.layers['component1'].keys():
            layer_info = self.layers['component1'][layer_name]
            layer = add_layer(component1, layer_name, layer_info)
            z_origin += layer_info[-1]
            if "interposer" in layer_name.lower():
                layer.attrib["MaxDepth"] = str(self.max_depth)
                if os.path.exists(os.path.join(thermal_dir, "intp_flp.csv")):
                    ET.SubElement(layer, "Floorplan", attrib={
                        "Type": "TSV", "File": str(os.path.join(thermal_dir, "intp_flp.csv"))
                    })
        self.z_origin = z_origin
        # Active component
        active = ET.SubElement(package, "Component", attrib={
            "number": "2",
            "type": "Active",
            "FillingMaterial": "Epoxy"
        })
        ET.SubElement(active, "Geometry", attrib={
            "Length": str(self.intp_width/1e3),
            "Width": str(self.intp_height/1e3),
            "XYorigin": "Origin"
        })

        # Add chiplets to the Active component
        for chiplet_idx in range(self.num_chiplets):
            chiplet = ET.SubElement(active, "Chiplet", attrib={
                "name": f"Chiplet_{chiplet_idx}",
                "FillingMaterial": "Si"
            })
            ET.SubElement(chiplet, "Geometry", attrib={
                "Length": str(self.width[chiplet_idx]/1e3),
                "Width": str(self.height[chiplet_idx]/1e3),
                "XYorigin": f"({self.x[chiplet_idx]/1e3}, {self.y[chiplet_idx]/1e3})"
            })
            for layer_name in self.layers['component2'].keys():
                layer = add_layer(chiplet, layer_name, self.layers['component2'][layer_name])
                if "act" in layer_name.lower():
                    ET.SubElement(layer, "Floorplan", attrib={
                        "Type": "Macro", 
                        "File": os.path.join(thermal_dir, f"{dir_name}/flp/Chiplet_{chiplet_idx}_flp.csv")
                    })
                    ET.SubElement(layer, "Power", attrib={
                        "File": os.path.join(thermal_dir, f"{dir_name}/power/Chiplet_{chiplet_idx}_power.csv")
                    })
        
        tim1 = ET.SubElement(package, "Component", attrib={
            "number": "3", 
            "type": "Bulk"
        })
        ET.SubElement(tim1, "Geometry", attrib={
            "Length": str(self.intp_width/1e3),
            "Width": str(self.intp_height/1e3),
            "XYorigin": "Center"
        })
        for layer_name in self.layers['component3'].keys():
            add_layer(tim1, layer_name, self.layers['component3'][layer_name])
        
        sp_info = self.layers["Spreader"]
        spreader = ET.SubElement(package, "Component", attrib={
            "number": "4", 
            "type": "Sink", 
            "sinktype": "spreader"
        })
        ET.SubElement(spreader, "Geometry", attrib={
            "Length": str(sp_info[2]),
            "Width": str(sp_info[3]),
            "XYorigin": "Center"
        })
        layer_sp = ET.SubElement(spreader, "Layer", attrib={
            "name": "SPREADER",
            "FillingMaterial": sp_info[0],
        })
        ET.SubElement(layer_sp, "Geometry", attrib={
            "Thickness": str(sp_info[1])
        })
        
        sk_info = self.layers["Sink"]
        sink = ET.SubElement(package, "Component", attrib={
            "number": "5", 
            "type": "Sink", 
            "sinktype": "nopackage"
        })
        ET.SubElement(sink, "Geometry", attrib={
            "Length": str(sk_info[2]),
            "Width": str(sk_info[3]),
            "XYorigin": "Center"
        })
        layer_sk = ET.SubElement(sink, "Layer", attrib={
            "name": "SINK",
            "FillingMaterial": sk_info[0],
        })
        ET.SubElement(layer_sk, "Geometry", attrib={
            "Thickness": str(sk_info[1])
        })
        
        # Convert the XML tree to a string and format it
        xml_string = ET.tostring(package, encoding='utf-8', method='xml')
        pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="    ")
        return pretty_xml
    
    def gen_flp_and_power(self, dir_name, alpha=0.8, T_base=85, gamma=0.015):
        library = gdspy.GdsLibrary()
        layer = 1
        datatype = 0
        with open(os.path.join(self.path, f"powerlib.yml")) as f:
            powerlib = yaml.safe_load(f)  # Using safe_load instead of load
    
        os.makedirs(os.path.join(self.path, f"{dir_name}/flp/"), exist_ok=True)
        os.makedirs(os.path.join(self.path, f"{dir_name}/power/"), exist_ok=True)
        for idx in range(self.num_chiplets):
            flp_filename = os.path.join(self.path, f"{dir_name}/flp/Chiplet_{idx}_flp.csv")
            gds_filename = os.path.join(self.path, f"{dir_name}/flp/Chiplet_{idx}_flp.gds")
            power_filename = os.path.join(self.path, f"{dir_name}/power/Chiplet_{idx}_power.csv")
            name, x, y, length, width = f'Chiplet{idx}', 0, 0, self.width[idx]/1e3,self.height[idx]/1e3
            
            with open(flp_filename, 'w', newline='') as flp_file:
                writer = csv.writer(flp_file)
                writer.writerow(['UnitName', 'X', 'Y', 'Z', 'Length', 'Width','Thickness','Label'])
                writer.writerow([name,x,y,0,length,width,'',''])

            #floorplan_data = pd.read_csv(flp_filename)
            if name in gdspy.current_library.cells:
                del gdspy.current_library.cells[name]
            cell = library.new_cell(name)
            for _ in range(1):
                rectangle = gdspy.Rectangle(
                    (x*1e6, y*1e6), ((x+length)*1e6, (y+width)*1e6), layer=layer, datatype=datatype
                )
                cell.add(rectangle)
            library.write_gds(gds_filename)
            power_dyn, power_leak = float(self.power[idx] * alpha), float(self.power[idx] * (1-alpha))

            powerlib['cells'][f'Chiplet{idx}'] = {
                "internal":{
                    "type": "constant", "data": power_dyn
                },
                "leakage":{
                    "type": "exp", "data": f"{power_leak:.6f} {T_base} {gamma}"
                },
            }

            with open(power_filename, 'w', newline='') as power_file:
                writer = csv.writer(power_file)
                writer.writerow([f'Chiplet{idx}'])
                writer.writerow(["1"])
                #writer.writerow(['UnitName', 'Power_dyn', 'Power_leak'])
                #writer.writerow([f'Chiplet{idx}', self.power[idx] * alpha, self.power[idx] * (1 - alpha)])

        # Write the modified powerlib back to file
        with open(os.path.join(self.path, f"{dir_name}/powerlib.yml"), 'w') as f:
            yaml.dump(powerlib, f)
        
    def run(self, dir_name, default=0):
        os.system('rm ' + self.path + '{*.xml,*.steady}')
        os.makedirs(os.path.join(self.path, f"{dir_name}"), exist_ok=True)
        pretty_xml = self.gen_stack(dir_name)
        
        # Write the formatted XML to the specified file
        xml_file = os.path.join(self.path, f"{dir_name}/package.xml")
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        self.gen_flp_and_power(dir_name)
        cmd = ["python3", "/home/qpwang/ATSim3D/srcv2/ATSim3_5D.py", 
               "-xml", xml_file, 
               "-config", os.path.join(self.path, "sim.config"),
               "--output_path",os.path.join(self.path, f"{dir_name}/")]

        if default:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr = subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
        else:
            os.system(" ".join(cmd))
    
    def getres(self, dir_name):
        res = np.loadtxt(os.path.join(self.path, f"{dir_name}/raw/steadytemp.txt"))
        z_min = self.z_origin
        z_max = self.z_origin+1e-6
        plane_idx = np.where((res[:,4]>=z_min) & (res[:,4]<=z_max))[0]
        x = (res[plane_idx,0]+res[plane_idx,1])/2
        y = (res[plane_idx,2]+res[plane_idx,3])/2
        val = res[plane_idx,-1]
        xi, yi = np.meshgrid(
            np.linspace(0, self.intp_width/1e3, self.num_grid_x), 
            np.linspace(0, self.intp_height/1e3, self.num_grid_y)
        )
        zi = griddata((x,y),val,(xi,yi),method='cubic')
        print(f"X range: {x.min():.3f}, {x.max():.3f}, Y: {y.min():.3f}, {y.max():.3f}, Zvals: {np.unique(res[plane_idx,4])}~{np.unique(res[plane_idx,5])}") 
        nan_mask = np.isnan(zi)
        if np.any(nan_mask):
            print(f"Nan vals in z: {nan_mask.sum()}/{nan_mask.size}")
            zi[nan_mask] = griddata((x,y),val,(xi,yi),method='linear')[nan_mask]
            nan_mask = np.isnan(zi)
            print(f"Nan vals in z: {nan_mask.sum()}/{nan_mask.size}")
            zi[nan_mask] = griddata((x,y),val,(xi,yi),method='nearest')[nan_mask]

        return np.transpose(zi,(1,0))
        

class Warpage_solver(ATSim_solver):
    def __init__(self, system):
        super(Warpage_solver, self).__init__(system, 1, system.thermal_dir)
        
        
    def run(self, dir_name, default=0):
        os.system('rm ' + self.path + '{*.xml,*.steady}')
        os.makedirs(os.path.join(self.path, f"{dir_name}"), exist_ok=True)
        pretty_xml = self.gen_stack(dir_name)
        
        # Write the formatted XML to the specified file
        xml_file = os.path.join(self.path, f"{dir_name}/package.xml")
        sim_config = os.path.join(self.path, "sim.config")
        output_dir = os.path.join(self.path, f"{dir_name}/")
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        self.gen_flp_and_power(dir_name)
        
        
        cmd = ["python3", "/home/qpwang/ATSim3D/srcv2/ATSim3_5D.py", 
               "-xml", xml_file, "-config", sim_config, "--output_path", output_dir]

        if default:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr = subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
        else:
            os.system(" ".join(cmd))
    
        cmd = ["python3", "/home/qpwang/ATSim3D-warpage/srcv1/Main.py", 
               "-xml", xml_file, "-config", sim_config, "--output_path", output_dir]

        if default:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr = subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
        else:
            os.system(" ".join(cmd))
    
    def getres(self, dir_name):
        res = np.loadtxt(os.path.join(self.path, f"{dir_name}/raw/steadytemp.txt"))
        #print(np.unique(res[:,4]), np.unique(res[:,5]))
        z_min = self.z_origin
        z_max = self.z_origin+1e-6
        plane_idx = np.where((res[:,4]>=z_min) & (res[:,4]<=z_max))[0]
        x = (res[plane_idx,0]+res[plane_idx,1])/2
        y = (res[plane_idx,2]+res[plane_idx,3])/2
        val = res[plane_idx,-1]
        xi, yi = np.meshgrid(
            np.linspace(0, self.intp_width/1e3, self.num_grid_x), 
            np.linspace(0, self.intp_height/1e3, self.num_grid_y)
        )
        zi = griddata((x,y),val,(xi,yi),method='cubic')
        print(f"X range: {x.min():.3f}, {x.max():.3f}, Y: {y.min():.3f}, {y.max():.3f}, Zvals: {np.unique(res[plane_idx,4])}~{np.unique(res[plane_idx,5])}") 
        nan_mask = np.isnan(zi)
        if np.any(nan_mask):
            print(f"Nan vals in z: {nan_mask.sum()}/{nan_mask.size}")
            zi[nan_mask] = griddata((x,y),val,(xi,yi),method='linear')[nan_mask]
            nan_mask = np.isnan(zi)
            print(f"Nan vals in z: {nan_mask.sum()}/{nan_mask.size}")
            zi[nan_mask] = griddata((x,y),val,(xi,yi),method='nearest')[nan_mask]

        temp_map = np.transpose(zi,(1,0))
        res = np.loadtxt(os.path.join(self.path, f"{dir_name}/raw/disp_z.txt"))
        warpage = np.reshape(res, (100,100))
        return temp_map, warpage

        
class HotSpot_solver(Thermal_solver):
    def __init__(self, system):
        super(HotSpot_solver, self).__init__(system, 3, system.thermal_dir)
        
    def clean_hotspot(self):
        os.system('rm ' + self.path + '{*.flp,*.lcf,*.ptrace,*.steady}')
        os.system('rm ' + self.path + 'new_hotspot.config')

    def gen_flp(self, filename):
        # material properties
        UnderFill = "\t2.32E+06\t0.625\n"
        Copper = "\t3494400\t0.0025\n"
        Silicon = "\t1.75E+06\t0.01\n"
        resistivity_Cu, specHeat_Cu = 0.0025, 3494400
        resistivity_UF, specHeat_UF = 0.625, 2320000
        resistivity_Si, specHeat_Si = 0.01, 1750000
        C4_diameter, C4_edge 	= 0.000250, 0.000600
        TSV_diameter, TSV_edge  = 0.000010, 0.000050  
        ubump_diameter, ubump_edge = 0.000025, 0.000045
        
        Aratio_C4 = (C4_edge/C4_diameter)*(C4_edge/C4_diameter)-1			# ratio of white area and C4 area
        Aratio_TSV= (TSV_edge/TSV_diameter)*(TSV_edge/TSV_diameter)-1
        Aratio_ubump=(ubump_edge/ubump_diameter)*(ubump_edge/ubump_diameter)-1
        resistivity_C4=(1+Aratio_C4)*resistivity_Cu*resistivity_UF/(resistivity_UF+Aratio_C4*resistivity_Cu)
        resistivity_TSV=(1+Aratio_TSV)*resistivity_Cu*resistivity_Si/(resistivity_Si+Aratio_TSV*resistivity_Cu)
        resistivity_ubump=(1+Aratio_ubump)*resistivity_Cu*resistivity_UF/(resistivity_UF+Aratio_ubump*resistivity_Cu)
        specHeat_C4=(specHeat_Cu+Aratio_C4*specHeat_UF)/(1+Aratio_C4)
        specHeat_TSV=(specHeat_Cu+Aratio_TSV*specHeat_Si)/(1+Aratio_TSV)
        specHeat_ubump=(specHeat_Cu+Aratio_ubump*specHeat_UF)/(1+Aratio_ubump)
        mat_C4 = "\t"+str(specHeat_C4)+"\t"+str(resistivity_C4)+"\n"
        mat_TSV = "\t"+str(specHeat_TSV)+"\t"+str(resistivity_TSV)+"\n"
        mat_ubump = "\t"+str(specHeat_ubump)+"\t"+str(resistivity_ubump)+"\n"
        Head_description = "# Line Format: <unit-name>\\t<width>\\t<height>\\t<left-x>\\t<bottom-y>\\t"+\
                            "[<specific-heat>]\\t[<resistivity>]\n"+"# all dimensions are in meters\n"+\
                            "# comment lines begin with a '#' \n"+"# comments and empty lines are ignored\n\n"

        with open(self.path + filename+ 'L0_Substrate.flp','w') as L0_Substrate:
            L0_Substrate.write("# Floorplan for Substrate Layer with size "+\
                               str(self.intp_width/1000)+"x"+str(self.intp_height/1000)+" m\n")
            L0_Substrate.write(Head_description)
            L0_Substrate.write("Substrate\t"+str(self.intp_width/1000)+"\t"+str(self.intp_height/1000)+"\t0.0\t0.0\n")

        with open(self.path+filename +'L1_C4Layer.flp','w') as L1_C4Layer:
            L1_C4Layer.write("# Floorplan for C4 Layer \n")
            L1_C4Layer.write(Head_description)
            L1_C4Layer.write("C4Layer\t"+str(self.intp_width/1000)+"\t"+str(self.intp_height/1000)+"\t0.0\t0.0"+mat_C4)

        with open(self.path+filename +'L2_Interposer.flp','w') as L2_Interposer:
            L2_Interposer.write("# Floorplan for Silicon Interposer Layer\n")
            L2_Interposer.write(Head_description)
            L2_Interposer.write("Interposer\t"+str(self.intp_width/1000)+"\t"+str(self.intp_height/1000)+"\t0.0\t0.0"+mat_TSV)

        with open(self.path+filename + 'sim.flp','w') as SIMP:
            with open(self.path + filename + 'L3.flp', 'w') as L3_UbumpLayer:
                with open(self.path + filename + 'L4.flp', 'w') as L4_ChipLayer:
                    L3_UbumpLayer.write("# Floorplan for Microbump Layer \n")
                    L3_UbumpLayer.write(Head_description)
                    L4_ChipLayer.write("# Floorplan for Chip Layer\n")
                    L4_ChipLayer.write(Head_description)
                    L3_UbumpLayer.write('Edge_0\t' + str(self.intp_width/1000 - self.granularity/1000) + '\t' +\
                                        str(self.granularity/2/1000) + '\t'+str(self.granularity/2/1000)+'\t0\t' + mat_ubump)
                    L3_UbumpLayer.write('Edge_1\t' + str(self.intp_width/1000 - self.granularity/1000) + '\t' +\
                                        str(self.granularity/2/1000) + '\t'+str(self.granularity/2/1000)+'\t'+\
                                        str(self.intp_height/1000 - self.granularity/2/1000)+'\t' + mat_ubump)
                    L3_UbumpLayer.write('Edge_2\t' + str(self.granularity/2/1000) + '\t' + str(self.intp_height/1000) +\
                                        '\t0\t0\t' + mat_ubump)
                    L3_UbumpLayer.write('Edge_3\t' + str(self.granularity/2/1000) + '\t' + str(self.intp_height/1000) +\
                                        '\t'+str(self.intp_width/1000-self.granularity/2/1000)+'\t0\t' + mat_ubump)
                    L4_ChipLayer.write('Edge_0\t' + str(self.intp_width/1000 - self.granularity/1000) + '\t' +\
                                        str(self.granularity/2/1000) + '\t'+str(self.granularity/2/1000)+'\t0\t' + mat_ubump)
                    L4_ChipLayer.write('Edge_1\t' + str(self.intp_width/1000 - self.granularity/1000) + '\t' +\
                                        str(self.granularity/2/1000) + '\t'+str(self.granularity/2/1000)+'\t'+\
                                        str(self.intp_height/1000 - self.granularity/2/1000)+'\t' + mat_ubump)
                    L4_ChipLayer.write('Edge_2\t' + str(self.granularity/2/1000) + '\t' + str(self.intp_height/1000) +\
                                        '\t0\t0\t' + mat_ubump)
                    L4_ChipLayer.write('Edge_3\t' + str(self.granularity/2/1000) + '\t' + str(self.intp_height/1000) +\
                                        '\t'+str(self.intp_width/1000-self.granularity/2/1000)+'\t0\t' + mat_ubump)

                    x_offset0, y_offset0 = self.granularity / 2 / 1000, self.granularity / 2 / 1000
                    for i in range(0, len(self.x)):
                        x_offset1 = self.x[i] / 1000 #- self.width[i] / 1000 * 0.5
                        y_offset1 = self.y[i] / 1000 #- self.height[i] / 1000 * 0.5
                        L3_UbumpLayer.write("Chiplet_"+str(i)+"\t"+str(self.width[i]/1000)+"\t"+\
                                            str(self.height[i]/1000)+"\t"+str(x_offset1)+"\t"+str(y_offset1)+mat_ubump)
                        L4_ChipLayer.write("Chiplet_"+str(i)+"\t"+str(self.width[i]/1000)+"\t"+\
                                           str(self.height[i]/1000)+"\t"+str(x_offset1)+"\t"+str(y_offset1)+Silicon)
                        SIMP.write("Unit_"+str(i)+"\t"+str(self.width[i]/1000)+"\t"+\
                                   str(self.height[i]/1000)+"\t"+str(x_offset1)+"\t"+str(y_offset1)+"\n")
                        
        utils.fill_space.fill_space(x_offset0, self.intp_width/1000 - x_offset0, y_offset0, self.intp_height/1000 - y_offset0,
                                   self.path+filename+'sim', self.path+filename+'L3', self.path+filename+'L3_UbumpLayer', 
                                    UnderFill)
        utils.fill_space.fill_space(x_offset0, self.intp_width/1000 - x_offset0, y_offset0, self.intp_height/1000 - y_offset0,
                                   self.path+filename+'sim', self.path+filename+'L4', self.path+filename+'L4_ChipLayer', 
                                   UnderFill)
        
        with open(self.path+filename +'L5_TIM.flp','w') as L5_TIM:
            L5_TIM.write("# Floorplan for TIM Layer \n")
            L5_TIM.write(Head_description)
            L5_TIM.write("TIM\t"+str(self.intp_width/1000)+"\t"+str(self.intp_height/1000)+"\t0.0\t0.0\n")

        with open(self.path+filename + 'layers.lcf','w') as LCF:
            LCF.write("# File Format:\n")
            LCF.write("#<Layer Number>\n")
            LCF.write("#<Lateral heat flow Y/N?>\n")
            LCF.write("#<Power Dissipation Y/N?>\n")
            LCF.write("#<Specific heat capacity in J/(m^3K)>\n")
            LCF.write("#<Resistivity in (m-K)/W>\n")
            LCF.write("#<Thickness in m>\n")
            LCF.write("#<floorplan file>\n")
            LCF.write("\n# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n"+self.path+filename+"L0_Substrate.flp\n")
            LCF.write("\n# Layer 1: Epoxy SiO2 underfill with C4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n"+self.path+filename+"L1_C4Layer.flp\n")
            LCF.write("\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n"+self.path+filename+"L2_Interposer.flp\n")
            LCF.write("\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n"+self.path+filename+"L3_UbumpLayer.flp\n")
            LCF.write("\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n"+self.path+filename+"L4_ChipLayer.flp\n")
            LCF.write("\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n"+self.path+filename+"L5_TIM.flp\n")

        #if not os.path.isfile(self.path + 'new_hotspot.config'):
        with open(self.path + 'hotspot.config','r') as Config_in:
            with open(self.path + 'new_hotspot.config','w') as Config_out:
                size_spreader = (self.intp_width + self.intp_height) / 1000
                size_heatsink = 2 * size_spreader
                r_convec =  0.1 * 0.06 * 0.06 / size_heatsink / size_heatsink   #0.06*0.06 by default config
                for line in Config_in:
                    if 's_sink' in line:
                        Config_out.write(line.replace('0.06',str(size_heatsink)))
                    elif 's_spreader' in line:
                        Config_out.write(line.replace('0.03',str(size_spreader)))
                    elif line == '		-r_convec			0.1\n':
                        Config_out.write(line.replace('0.1',str(r_convec)))
                    else:
                        Config_out.write(line)
        
    def gen_ptrace(self, filename):
        num_component = 0
        component, component_name, component_index = [], [], []
        # Read components from flp file
        with open (self.path + filename + 'L4_ChipLayer.flp','r') as FLP:
            for line in FLP:
                line_sp = line.split()
                if line_sp:
                    if line_sp[0] != '#':
                        component.append(line_sp[0])
                        comp = component[num_component].split('_')
                        component_name.append(comp[0])
                        component_index.append(int(comp[1]))
                        num_component+=1

        with open (self.path + filename + '.ptrace','w') as Ptrace:
            # Write ptrace header
            for i in range(0,num_component):
                Ptrace.write(component[i]+'\t')
            Ptrace.write('\n')
            for i in range(0,num_component):
                if component_name[i] == 'Chiplet':
                    Ptrace.write(str(self.power[component_index[i]])+'\t')
                else:
                    Ptrace.write('0\t')
            Ptrace.write('\n')

    def run(self, filename, default=1):
        self.clean_hotspot()
        self.gen_flp(filename)
        self.gen_ptrace(filename)
        cmd = [self.path+"hotspot", "-c",self.path+"new_hotspot.config", 
               "-f",self.path+filename+"L4_ChipLayer.flp", 
               "-p",self.path+filename+".ptrace", 
               "-steady_file",self.path+filename+".steady", 
               "-grid_steady_file",self.path+filename+".grid.steady",
               "-model_type","grid", "-detailed_3D","on", 
               "-grid_layer_file",self.path+filename+"layers.lcf"]
        t1 = time.time()
        if default:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr = subprocess.PIPE)
            stdout, stderr = proc.communicate()
        else:
            os.system(" ".join(cmd))
            print(time.time()-t1)
            #outlist = stdout.split()
            #return stdout #(max(list(map(float,outlist[3::2])))-273.15)
        
    def getres(self, temp_file_name):
        temp_single = np.loadtxt(
            self.path+temp_file_name+".grid.steady"
        )[:,1].reshape(self.num_grid_x, self.num_grid_y)
        temp_single = np.transpose(temp_single,(1,0))[:,::-1]
        return temp_single