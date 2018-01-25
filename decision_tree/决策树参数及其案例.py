# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 09:08:17 2017

@author: pengb
需要将源代码中的remove(archive_path)注释掉
需要安装python插件pydotplus
需要下载GraphViz软件，并将bin目录添加至系统path
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

#生成决策树用graphviz画出来
housing = fetch_california_housing()
dtr = tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(housing.data[:, [6, 7]], housing.target)

dot_data = tree.export_graphviz(dtr, out_file=None, 
                                feature_names=housing.feature_names[6:8],
                                #['Latitude', 'Longitude']
                                filled = True,
                                impurity=False,
                                rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#FFF2DD')
Image(graph.create_png())
graph.write_png("决策树可视化.png")

data_train, data_test, target_train, target_test = train_test_split(housing.data,
                                                                    housing.target,
                                                                    test_size=0.2,
                                                                    random_state=42)
dtr = tree.DecisionTreeRegressor(random_state=42)
dtr.fit(data_train, target_train)
print(dtr.score(data_test, target_test))

rfr = RandomForestRegressor(random_state=42)
rfr.fit(data_train, target_train)
print(rfr.score(data_test, target_test))

tree_param_grid = {'min_samples_split':list((3, 6, 9)),
                   'n_estimators':list((10, 50, 100))}
#自动选择参数
grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print("grid_scores: ", grid.grid_scores_)
print("best_params: ", grid.best_params_)
print("best_score: ",  grid.best_score_)