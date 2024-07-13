''' these is the dump from pasthistory.ipynb file.'''

# using minx and maxx, i.e the boundary poits for the grid.
# initialize an empty list to store x-values for the custom grid
x_values = []
# generate x-values
current_x = minx
increment = 0.25 #this is an inceremnet of 0.25
while current_x <= maxx:
    x_values.append(current_x)  
    current_x += increment
len(x_values)

# using miny and maxy, i.e the boundary poits for the grid.
# initialize an empty list to store y-values for the custom grid
y_values = []
# generate y-values
current_y = miny
increment = 0.25 #this is an inceremnt of 0.25
while current_y <= maxy:
    y_values.append(current_y)  
    current_y += increment
len(y_values)

x_values = np.array(x_values, dtype=float) # x_values are the longitudes and y_values are the latitudes.
y_values = np.array(y_values, dtype=float)

if __name__ == "__main__":
    grid_cell = initialize_and_populate_grid(x_values, y_values)

grid_cell

xc=[i['Xc']for i in grid_cell]
yc=[i['Yc']for i in grid_cell]
x,y=np.meshgrid(xc,yc)
rectangle = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='epsg:4326')
fig, ax = plt.subplots()
ax.scatter(x, y, s=1, color='blue')
rectangle.plot(ax=ax, color='none', edgecolor='red')
plt.show()

def res_change(minx,miny,maxx,maxy,input_res,output_res):  
    
        # using minx and maxx, i.e the boundary poits for the grid.
    # initialize an empty list to store x-values for the custom grid
    x_values = []
    # generate x-values
    current_x = minx
    increment = input_res #this is an inceremnet of 0.25
    while current_x <= maxx:
        x_values.append(current_x)  
        current_x += increment
    len(x_values)
    # using miny and maxy, i.e the boundary poits for the grid.
    # initialize an empty list to store y-values for the custom grid
    y_values = []
    # generate y-values
    current_y = miny
    increment = input_res #this is an inceremnt of 0.25
    while current_y <= maxy:
        y_values.append(current_y)  
        current_y += increment
    len(y_values)
    x_values = np.array(x_values, dtype=float) # x_values are the longitudes and y_values are the latitudes.
    y_values = np.array(y_values, dtype=float)
    if __name__ == "__main__":
     grid_cell = initialize_and_populate_grid(x_values, y_values)
    xc=[i['Xc']for i in grid_cell]
    yc=[i['Yc']for i in grid_cell]
    x,y=np.meshgrid(xc,yc)
    rectangle = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='epsg:4326')
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1, color='blue')
    rectangle.plot(ax=ax, color='none', edgecolor='red')
    #ax.title('image for input resolution')
    plt.show()


    x_values_output = []
    # generate x-values
    current_x_output = minx
    increment = output_res #this is an inceremnet of 0.25
    while current_x_output <= maxx:
        x_values_output.append(current_x_output)  
        current_x += increment
    len(x_values_output)
    # using miny and maxy, i.e the boundary poits for the grid.
    # initialize an empty list to store y-values for the custom grid
    y_values_output = []
    # generate y-values
    current_y_output = miny
    increment = output_res #this is an inceremnt of 0.25
    while current_y_output <= maxy:
        y_values_output.append(current_y_output)  
        current_y_output += increment
    len(y_values_output)
    x_values_output = np.array(x_values_output, dtype=float) # x_values are the longitudes and y_values are the latitudes.
    y_values_output = np.array(y_values_output, dtype=float)
    if __name__ == "__main__":
     grid_cell_output = initialize_and_populate_grid(x_values_output, y_values_output)
    xc_output=[i['Xc']for i in grid_cell_output]
    yc_output=[i['Yc']for i in grid_cell_output]
    x_out,y_out=np.meshgrid(xc_output,yc_output)
    rectangle_out = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs='epsg:4326')
    fig_out, ax_out = plt.subplots()
    ax_out.scatter(x_out, y_out, s=1, color='black')
    rectangle_out.plot(ax=ax, color='none', edgecolor='red')
    #ax_out.title('image for output resolution')
    plt.show()

if __name__ == "__main__":
    res_change(minx, miny, maxx, maxy, 0.25, 0.1)


# a small trick i wa trying to do, to get all the data for all the years together.

#repeating the lat, lon for 40 years.
start_date = '1984-01-01'
end_date = '2023-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
dates_df = pd.DataFrame({'date': date_range})
n_dates = len(date_range)
df_rep = pd.concat([inter_points]*n_dates, ignore_index=True)
df_rep['date']=np.tile(date_range, len(inter_points))
df_rep = df_rep[['date', 'lon', 'lat']]
df_rep


# interpolation using a diffrent class from scipy

def interpolate(initial_res_points, data, coords, method_used):
    # Ensure the entered coordinates have accuracy of only 1 digit after the decimal.
    if method_used in ['linear', 'nearest', 'cubic']:
        interpolator = sci.RegularGridInterpolator(initial_res_points, data, method=method_used)
        return interpolator(coords)
    else:
        raise ValueError("Invalid method. Supported methods are 'linear', 'nearest', and 'cubic'.")
    
# from interpolating.py
def save_images(self, interpolated_data, output_folder):
    '''this method will save the images in the output folder'''
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each time step and save the image
    for i, data in tqdm(enumerate(interpolated_data), total=len(interpolated_data)):
        # Normalize the data to be in the range 0-255
        data_normalized = (255 * (data - np.min(data)) / (np.ptp(data))).astype(np.uint8)
        
        # Create an Image object from the data
        image = PIL.Image.fromarray(data_normalized)
        #showing the image
        image
        # Save the image
        image.save(os.path.join(output_folder, f'image_{i}.png'))