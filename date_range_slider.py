cols1=df.loc[:, ['country','date', 'confirmed']]
cols2 = cols1[cols1['country'] == 'Afghanistan' ]


Overall = ColumnDataSource(data=cols1)
Curr=ColumnDataSource(data=cols2)
#plot and the menu is linked with each other by this callback function
callback = CustomJS(args=dict(source=Overall, sc=Curr), code="""
var f = cb_obj.value
sc.data['date']=[]
sc.data['confirmed']=[]
for(var i = 0; i <= source.get_length(); i++){
	if (source.data['country'][i] == f){
		sc.data['date'].push(source.data['date'][i])
		sc.data['confirmed'].push(source.data['confirmed'][i])
	 }
}   
   
sc.change.emit();
""")
menu = Select(options=country_list,value='Afghanistan', title = 'Country')  # drop down menu
bokeh_p=figure(x_axis_label ='date', y_axis_label = 'confirmed', y_axis_type="linear",x_axis_type="datetime") #creating figure object 
bokeh_p.line(x='date', y='confirmed', color='green', source=Curr) # plotting the data using glyph circle
menu.js_on_change('value', callback) # calling the function on change of selection
date_range_slider = DateRangeSlider(value=(min(df['date']), max(df['date'])),
                                    start=min(df['date']), end=max(df['date']))
date_range_slider.js_link("value", bokeh_p.x_range, "start", attr_selector=0)
date_range_slider.js_link("value", bokeh_p.x_range, "end", attr_selector=1)
layout = column(menu, date_range_slider, bokeh_p)
show(layout) # displaying the layout