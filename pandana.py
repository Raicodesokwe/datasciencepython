import pandas as pd

kamnyweso=({
    "white rum":[23,4,57],
    'red rum':[12,4,34]
})

sales=pd.DataFrame(kamnyweso,index=["harry","bazu","zindi"])
sales['Henezii']=[99,67,34]
print(sales)