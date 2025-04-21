import os
import time
import subprocess

import numpy as np

import utils.fill_space

class Thermal_solver():
    def __init__(self, thermal_root_path):
        self.path = thermal_root_path

    def set_params(self, system):
        '''
        All the dimensions in the Thermal_solver are mm
        '''
        self.num_chiplets = system.num_chiplets
        self.num_nodes = system.num_nodes
        self.granularity = system.granularity/1e3
        self.decimal = 6 #len(str(self.granularity).split('.')[-1])
        self.intp_width = np.round((system.intp_width)/1e3, self.decimal+1)
        self.intp_height = np.round((system.intp_height)/1e3, self.decimal+1)

    def set_pos(self, power, pos, width, height):
        num_chiplets = self.num_chiplets
        num_nodes = self.num_nodes
        self.power = power
        self.x = np.round(pos[0]/1e3, self.decimal+1)
        self.y = np.round(pos[1]/1e3, self.decimal+1) #//self.granularity*self.granularity
        self.width = np.round(width/1e3, self.decimal+2) 
        self.height = np.round(height/1e3, self.decimal+2)

    def clean_hotspot(self, filename):
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
                        x_offset1 = self.x[i] / 1000 - self.width[i] / 1000 * 0.5
                        y_offset1 = self.y[i] / 1000 - self.height[i] / 1000 * 0.5
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

    def run_hotspot(self, filename, default=1):
        self.clean_hotspot(filename)
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
        
