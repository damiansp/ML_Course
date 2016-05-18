import graphlab
import matplotlib.pyplot as plt

sales = graphlab.SFrame('home_data.gl/')
#print sales

graphlab.canvas.set_target('ipynb')
sales.show(view = 'Scatter Plot', x = 'sqft_living', y = 'price')

train_data, test_data = sales.random_split(0.8, seed = 0)
sqft_model = graphlab.linear_regression.create(train_data,
                                               target = 'price',
                                               features = ['sqft_living'])
print sqft_model.evaluate(test_data)
# %matplotlib inline # in notebook prints output in notebook
plt.plot(test_data['sqft_living'],
         test_data['price'],
         '.',
         test_data['sqft_living'],
         sqft_model.predict(test_data),
         '-')

sqft_model.get('coefficients')

my_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
sales[my_features].show()
sales.show(view = 'BoxWhisker Plot', x = 'zipcode', y = 'price')

my_features_model = graphlab.linear_regression.create(
    train_data, target = 'price', features = my_features)

print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

house1 = sales[sales['id'] == '5309101200']
house1

#<img src="house-5309101200.jpg" />

print house1['price']
print sqft_model.predict(house1)
print my_features_model.predict(house1)

house2 = sales[slaes['id'] == '1925069082']
house2

print sqft_model.predict(house2)
print my_features_model.predict(house2)


bill_gates = {'bedrooms': [8],
              'bathrooms':[25],
              'sqft_living':[50000],
              'sqft_lot':[225000],
              'floors':[4],
              'zipcode':['98039'],
              'condition':[10],
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}

print sqft_model.predict(graphlab.SFrame(bill_gates))
print my_features_model.predict(graphlab.SFrame(bill_gates))
