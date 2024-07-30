'''
==========================================================================================================================
the methods in this file are :
1. calculate_grid_parameters (from interpolation_to_0.1.ipynb)
2. initialize_and_populate_grid (from interpolation_to_0.1.ipynb)
3. res_change (from interpolation_to_0.1.ipynb)
4. fitting_grid - this method takes the input of the basin file, and fits it in the grid made in read_shapefile
==========================================================================================================================
'''

import geopandas as gpd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
#import os
import shapely
import pandas as pd
from pyproj import Transformer

#print("the default epsg is 4326, change if required")
class Resize:
   # function to calculate grid parameters
    def calculate_grid_parameters(self, coord):
        d = np.diff(coord)[0] / 2
        cell_boundaries = np.concatenate([[coord[0] - d], coord + d])
        return cell_boundaries

    def initialize_and_populate_grid(self, x_values, y_values):
        ngrid_cell = len(x_values) * len(y_values)
        grid_cell = [{} for _ in range(ngrid_cell)]
        
        dy = y_values[1] - y_values[0]
        k = 0
        
        loncell = self.calculate_grid_parameters(x_values)
        latcell = self.calculate_grid_parameters(y_values)
        
        for i, lon in enumerate(x_values):
            for j, lat in enumerate(y_values):
                x_coords = [round(loncell[i], 2), round(loncell[i], 2), round(loncell[i+1], 2), round(loncell[i+1], 2), round(loncell[i], 2)]
                y_coords = [round(latcell[j], 2), round(latcell[j+1], 2), round(latcell[j+1], 2), round(latcell[j], 2), round(latcell[j], 2)] if dy > 0 else \
                        [round(latcell[j+1], 2), round(latcell[j], 2), round(latcell[j], 2), round(latcell[j+1], 2), round(latcell[j+1], 2)]
                
                grid_cell[k] = {'X':x_coords,'Y':y_coords,'Xc': round(lon, 2), 'Yc': round(lat, 2)}
                k += 1
        return grid_cell    
        
    def res_change(self,minx, miny, maxx, maxy, input_res, output_res,crs=4326):
        '''returns the x and y coordinates of centroids for the input and output resolution grid, in the order of 
        x,y being the x and y coordinates of the centroids of the grid cells for the input resolution grid and x_out,y_out being the x and y coordinates of the centroids of the grid cells for the output resolution grid.
        the function also returns the grid for output resolution.
        it also visualizes the grid for input and output resolution.'''
        
        if ((maxx-minx)*(maxy-miny)!=((input_res/output_res)**2)*(maxx-minx)*(maxy-miny)):
            print("The resolution change is not exact.")
            #back_calc((minx, miny, maxx, maxy))
            
        else:
            print("The resolution change is exact.")

        # Generate x-values for the input resolution grid (longitude)
        x_values = []
        current_x = minx
        while current_x < maxx:
            x_values.append(current_x)
            current_x += input_res
        
        # Generate y-values for the input resolution grid (latitude)
        y_values = []
        current_y = miny
        while current_y < maxy:
            y_values.append(current_y)
            current_y += input_res
        
        x_values = np.array(x_values, dtype=float)
        y_values = np.array(y_values, dtype=float)
        
        grid_cell = self.initialize_and_populate_grid(x_values, y_values)
        xc = [i['Xc'] for i in grid_cell]
        yc = [i['Yc'] for i in grid_cell]
        x, y = np.meshgrid(xc, yc)
        
        
        rectangle = gpd.GeoSeries([shapely.geometry.box(minx, miny, maxx, maxy)], crs=f'epsg:{crs}')
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1, color='blue')
        rectangle.plot(ax=ax, color='none', edgecolor='red')
        plt.title('Image for input resolution')
        plt.show()
        
        # Generate x-values for the output resolution grid
        x_values_output = []
        current_x_output = minx
        while current_x_output < maxx:
            x_values_output.append(current_x_output)
            current_x_output += output_res
        
        # Generate y-values for the output resolution grid
        y_values_output = []
        current_y_output = miny
        while current_y_output < maxy:
            y_values_output.append(current_y_output)
            current_y_output += output_res
        
        x_values_output = np.array(x_values_output, dtype=float)
        y_values_output = np.array(y_values_output, dtype=float)
        
        grid_cell_output = self.initialize_and_populate_grid(x_values_output, y_values_output)
        xc_output = [i['Xc'] for i in grid_cell_output]
        yc_output = [i['Yc'] for i in grid_cell_output]
        x_out, y_out = np.meshgrid(xc_output, yc_output)
        
        rectangle_out = gpd.GeoSeries([shapely.geometry.box(minx, miny, maxx, maxy)], crs=f'epsg:{crs}')
        fig_out, ax_out = plt.subplots()
        ax_out.scatter(x_out, y_out, s=1, color='black')
        rectangle_out.plot(ax=ax_out, color='none', edgecolor='red')
        plt.title('Image for output resolution')
        plt.show()
        print(f"dims of {output_res} grid= ",(((max(yc_output) - min(yc_output)) // output_res,(max(xc_output)- min(xc_output)) // output_res)))
        print(f"\ndims of {input_res} grid= ",(((max(yc)- min(yc)) //input_res,(max(xc) - min(xc)) //input_res)))
        extent_calculator = lambda crs_from, crs_to, coords: print(f"Extent in y and x (in km): {' '.join(str((max(c) - min(c))/1000) for c in zip(*Transformer.from_crs(crs_from, crs_to).transform(*zip(*coords))))}")
        extent_calculator(crs, 3857, [[miny, maxy], [minx, maxx]])
        #return x,y,x_out,y_out,grid_cell_output

    def res_change_general(self,minx, miny, maxx, maxy, input_res, output_res,crs=4326):

        if ((maxx-minx)*(maxy-miny)!=((input_res/output_res)**2)*(maxx-minx)*(maxy-miny)):
            print("The resolution change is not exact.")
            #back_calc((minx, miny, maxx, maxy))
            
        else:
            print("The resolution change is exact.")

        # Generate x-values for the input resolution grid (here 0.25 degree)
        x_values = []
        current_x = minx
        while current_x < maxx:
            x_values.append(current_x)
            current_x += input_res
        
        # Generate y-values for the input resolution grid (here 0.25 degree)
        y_values = []
        current_y = miny
        while current_y < maxy:
            y_values.append(current_y)
            current_y += input_res
        
        x_values = np.array(x_values, dtype=float)
        y_values = np.array(y_values, dtype=float)
        
        grid_cell = self.initialize_and_populate_grid(x_values, y_values)

        #making the meshgrid for the input resolution grid
        x = [i['Xc'] for i in grid_cell]
        y = [i['Yc'] for i in grid_cell]
        x, y = np.meshgrid(x, y)
        
        # Generate x-values for the output resolution grid (here 0.1 degree)
        x_values_output = []
        current_x_output = minx
        while current_x_output < maxx:
            x_values_output.append(current_x_output)
            current_x_output += output_res
        
        # Generate y-values for the output resolution grid (here 0.1 degree)
        y_values_output = []
        current_y_output = miny
        while current_y_output < maxy:
            y_values_output.append(current_y_output)
            current_y_output += output_res
        
        x_values_output = np.array(x_values_output, dtype=float)
        y_values_output = np.array(y_values_output, dtype=float)
        
        grid_cell_output = self.initialize_and_populate_grid(x_values_output, y_values_output)
        x_out = [i['Xc'] for i in grid_cell_output]
        y_out = [i['Yc'] for i in grid_cell_output]
        x_out, y_out = np.meshgrid(x_out, y_out)
        
        print("dims of grid= ",(((y_out[len(y_out)-1][0] - y_out[0][0]) // 0.1,(x_out[0][len(x_out[0])-1] - x_out[0][0]) // 0.1)))
        print("dims of grid= ",(((y[len(y)-1][0] - y[0][0]) //0.25,(x[0][len(x[0])-1] - x[0][0]) //0.25)))
        extent_calculator = lambda crs_from, crs_to, coords: print(f"Extent in y and x (in km): {' '.join(str((max(c) - min(c))/1000) for c in zip(*Transformer.from_crs(crs_from, crs_to).transform(*zip(*coords))))}")
        extent_calculator(crs, 3857, [[miny, maxy], [minx, maxx]])
        return np.meshgrid(x_values,y_values),np.meshgrid(x_values_output,y_values_output)