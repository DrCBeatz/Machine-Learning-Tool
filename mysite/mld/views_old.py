from django.http import HttpResponse
from django.urls import reverse_lazy, reverse
from django.views import View
from mld.models import ML_Type, Dataset, Algorithm, Entry, Graph, File_Type
from django.utils.decorators import method_decorator
from mld.owner import OwnerListView, OwnerDetailView, OwnerCreateView, OwnerUpdateView, OwnerDeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView
from django.db.models import Q
from mld.forms import CreateForm, CreateGraphForm, CreateDatasetForm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_curve, auc, classification_report, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import mglearn

from io import StringIO, BytesIO
from django.shortcuts import render, redirect, get_object_or_404
from django.template import loader
import time


cancer=load_breast_cancer()
# boston=mglearn.datasets.load_extended_boston()
boston=load_boston()

# bank = pd.read_csv("/home/drcbeatz/django_projects/mysite/mld/datasets/banknote_authentication.csv")
# bank_data = bank.iloc[:, 0:-1]
# bank_target = bank.iloc[:, -1]

# survival = pd.read_csv("/home/drcbeatz/django_projects/mysite/mld/datasets/haberman-data.csv")
# survival_data = survival.iloc[:,0:-1]
# survival_target = survival.iloc[:,-1]

# sonar = pd.read_csv("/home/drcbeatz/django_projects/mysite/mld/datasets/sonar.csv")
# sonar_data = sonar.iloc[:,0:-1]
# sonar_target = sonar.iloc[:, -1]

# ionosphere = pd.read_csv("/home/drcbeatz/django_projects/mysite/mld/datasets/ionosphere.csv")
# ionosphere_data = ionosphere.iloc[:,0:-1]
# ionosphere_target = ionosphere.iloc[:, -1]

# pima = pd.read_csv("/home/drcbeatz/django_projects/mysite/mld/datasets/pima-indians-diabetes.csv")
# pima_data = pima.iloc[:,0:-1]
# pima_target = pima.iloc[:, -1]

class MldHomeView(TemplateView):
    template_name = "mld/mld_home.html"

class MldAboutView(TemplateView):
    template_name = "mld/mld_about.html"

class MldListView(OwnerListView):
    model = Entry
    # By convention:
    template_name = "mld/mld_list.html"
    def get(self, request) :
        entry_list = Entry.objects.all()
        if request.user.is_superuser == True :
            ctx = {'entry_list' : entry_list}
        else:
            ctx = {'entry_list' : entry_list.filter(owner=request.user)}
        return render(request, self.template_name, ctx)

# def search(request):
#     status_list = Data.objects.all()
#     status_filter = status_list.filter(user=request.user) //get current user id
#     return render(request, 'users/data.html', {'filter': status_filter})

class GraphListView(OwnerListView):
    model = Graph
    # By convention:
    template_name = "mld/graph_list.html"
    def get(self, request) :
        graph_list = Graph.objects.all()
        if request.user.is_superuser == True :
            ctx = {'graph_list' : graph_list}
        else:
            ctx = {'graph_list' : graph_list.filter(owner=request.user)}
        return render(request, self.template_name, ctx)

class DatasetListView(OwnerListView):
    model = Dataset
    # By convention:
    template_name = "mld/dataset_list.html"
    def get(self, request) :
        dataset_list = Dataset.objects.all()
        if request.user.is_superuser == True :
            ctx = {'dataset_list' : dataset_list}
        else:
            ctx = {'dataset_list' : dataset_list.filter(Q(public=True) | Q(owner=request.user))}
        return render(request, self.template_name, ctx)

class AlgorithmListView(OwnerListView):
    model = Algorithm
    # By convention:
    template_name = "mld/algorithm_list.html"
    def get(self, request) :
        algorithm_list = Algorithm.objects.all()
        ctx = {'algorithm_list' : algorithm_list}
        return render(request, self.template_name, ctx)

class MldDetailView(OwnerDetailView):
    model = Entry
    template_name = "mld/mld_detail.html"
    def get(self, request, pk) :
        x = Entry.objects.get(id=pk)

        if x.dataset.target_datatype == 'int':
            target_data = np.frombuffer(x.target_data, dtype=np.uint64)
        elif x.dataset.target_datatype == 'float':
            target_data = np.frombuffer(x.target_data)

        target_data_length = len(target_data)
        # roc_auc = x.roc_auc
        # tpr = np.frombuffer(x.tpr)
        # fpr = np.frombuffer(x.fpr)
        # thresholds = np.frombuffer(x.thresholds)

        context = { 'entry' : x, 'target_data' : target_data, 'target_data_length' : target_data_length}
        # context['plot'] = return_roc_graph(x.algorithm.title, tpr, fpr, thresholds, roc_auc)
        return render(request, self.template_name, context)



class GraphDetailView(OwnerDetailView):
    model = Graph
    template_name = "mld/graph_detail.html"
    def get(self, request, pk) :
        x = Graph.objects.get(id=pk)

        # objects = Entry.objects.filter(query).select_related().order_by('-updated_at')[:10]
        # algorithm_name = Graph.algorithm.name
        context = { 'graph' : x }
        # if x.parameter_to_plot == 1:
        #     param_name = x.algorithm.parameter_1_name
        # elif x.parameter_to_plot == 2:
        #     param_name = x.algorithm.parameter_2_name
        # elif x.parameter_to_plot == 3:
        #     param_name = x.algorithm.parameter_3_name

        # context['plot'] = return_graph(x.dataset.id, x.algorithm.id, x.parameter_to_plot, param_name)

        return render(request, self.template_name, context)

def stream_file(request, pk):
    graph = get_object_or_404(Graph, id=pk)

    if graph.parameter_to_plot == 1:
        param_name = graph.algorithm.parameter_1_name
    elif graph.parameter_to_plot == 2:
        param_name = graph.algorithm.parameter_2_name
    elif graph.parameter_to_plot == 3:
        param_name = graph.algorithm.parameter_3_name

    response = HttpResponse()
    response['Content-Type'] = 'png'
    image = return_graph(graph.ml_type.id, graph.dataset.id, graph.algorithm.id, graph.parameter_to_plot, param_name)
    response['Content-Length'] = len(image)
    response.write(image)
    return response

def stream_roc_graph(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    tpr = np.frombuffer(entry.tpr)
    fpr = np.frombuffer(entry.fpr)
    thresholds = np.frombuffer(entry.thresholds)
    image = return_roc_graph(entry.algorithm.title, tpr, fpr, thresholds, entry.roc_auc)
    response['Content-Length'] = len(image)

    response.write(image)
    return response

def stream_precision_recall_graph(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    precision = np.frombuffer(entry.precision_c)
    recall = np.frombuffer(entry.recall_c)
    thresholds2 = np.frombuffer(entry.thresholds2)
    image = return_precision_recall_graph(entry.algorithm.title, precision, recall, thresholds2, entry.avg_precision)
    response['Content-Length'] = len(image)

    response.write(image)
    return response

def stream_dist_plot(request, pk):
    entry = get_object_or_404(Entry, id=pk)

    response = HttpResponse()
    response['Content-Type'] = 'png'

    if entry.dataset.target_datatype == 'int':
        target_data = np.frombuffer(entry.target_data, dtype=np.int64)
    elif entry.dataset.target_datatype == 'float':
        target_data = np.frombuffer(entry.target_data)

    # target_data = np.frombuffer(entry.target_data)

    image = return_dist_plot(target_data)
    response['Content-Length'] = len(image)
    response.write(image)
    return response

def return_dist_plot(target_data):

    # fig = plt.figure()
    # sns.set(rc={'figure.figsize':(11.7,8.27)})
    plt.clf()
    sns.displot(target_data, bins=15, kde=True, height=3)

    # sns.distplot(target_data, bins=30)

    # fig = sns.displot(target_data, bins=30)
    # plt.show()

    imgdata = BytesIO()
    # fig2 = fig.get_figure()
    # fig2.savefig(imgdata, format='png', dpi=300)
    plt.savefig(imgdata, format='png', dpi=300)
    # fig.savefig(imgdata, format='png', dpi=300)
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_roc_graph(algorithm_title, tpr, fpr, thresholds, roc_auc):
    fig = plt.figure()

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt_title = 'ROC Curve of ' + algorithm_title
    plt.title(plt_title)

    # fig.tight_layout()
    plt.show()

    # imgdata = StringIO()
    # fig.savefig(imgdata, format='svg')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', dpi=300)
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_precision_recall_graph(algorithm_title, precision, recall, thresholds, avg_precision):
    fig = plt.figure()

    plt.plot(precision, recall, 'b', label = 'Avg. Precision = %0.2f' % avg_precision)
    plt.legend(loc='best')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt_title = 'Precision-Recall Curve of ' + algorithm_title
    plt.title(plt_title)

    # fig.tight_layout()
    plt.show()

    # imgdata = StringIO()
    # fig.savefig(imgdata, format='svg')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', dpi=300)
    imgdata.seek(0)

    data = imgdata.getvalue()
    return data

def return_graph(ml_type_index, dataset_index, algorithm_index, parameter_number, parameter_name):

    if parameter_number == 1:
        entries = Entry.objects.filter(algorithm_id=algorithm_index, dataset_id=dataset_index).order_by('-parameter_1_value').values()
    elif parameter_number == 2:
        entries = Entry.objects.filter(algorithm_id=algorithm_index, dataset_id=dataset_index).order_by('-parameter_2_value').values()
    elif parameter_number == 3:
        entries = Entry.objects.filter(algorithm_id=algorithm_index, dataset_id=dataset_index).order_by('-parameter_3_value').values()

    training_accuracy = []
    test_accuracy = []
    r2_train_score = []
    r2_test_score = []
    parameter_values = []

    for item in entries:
        if parameter_number == 1:
            parameter_values.append(item['parameter_1_value'])
        elif parameter_number == 2:
            parameter_values.append(item['parameter_2_value'])
        elif parameter_number == 3:
            parameter_values.append(item['parameter_3_value'])

        if ml_type_index == 1:
            training_accuracy.append(item['train_accuracy'])
            test_accuracy.append(item['test_accuracy'])
        else:
            r2_train_score.append(item['r2_train_score'])
            r2_test_score.append(item['r2_test_score'])

    fig = plt.figure()

    if ml_type_index == 1:
        plt.plot(parameter_values, training_accuracy, label="training accuracy")
        plt.plot(parameter_values, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
    else:
        plt.plot(parameter_values, r2_train_score, label="R-squared train score")
        plt.plot(parameter_values, r2_test_score, label="R-squared test score")
        plt.ylabel("R-squared score")

    plt.xlabel(parameter_name)
    plt.legend()

    # imgdata = StringIO()
    # fig.savefig(imgdata, format='svg')

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png', dpi=300)

    imgdata.seek(0)

    data = imgdata.getvalue()
    return data


class DatasetDetailView(OwnerDetailView):
    model = Dataset
    template_name = "mld/dataset_detail.html"
    def get(self, request, pk) :
        x = Dataset.objects.get(id=pk)

        context = { 'dataset' : x }

        df = return_dataframe(x)

        ###
        # df_new = return_dataframe(obj.dataset)
        # s = df.iloc[:, -1]
        # if x.target_data != None:
        #     target_data = np.frombuffer(x.target_data, dtype=np.int64)

        html_data = df.to_html(classes='table table-responsive table-bordered table-hover table-condensed table-striped')

        context['html_data'] = html_data
        # context['target_data'] = target_data
        return render(request, self.template_name, context)

class AlgorithmDetailView(OwnerDetailView):
    model = Algorithm
    template_name = "mld/algorithm_detail.html"
    def get(self, request, pk) :
        x = Algorithm.objects.get(id=pk)

        context = { 'algorithm' : x }
        return render(request, self.template_name, context)

class MldCreateView(LoginRequiredMixin, View):
    template_name = 'mld/mld_form.html'
    success_url = reverse_lazy('mld:all')

    def get(self, request, pk=None):
        form = CreateForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        entry = form.save(commit=False)
        entry.owner = self.request.user
        entry = ml_train_test(entry)
        entry.save()
        return redirect(self.success_url)

class GraphCreateView(LoginRequiredMixin, View):
    template_name = 'mld/graph_form.html'
    success_url = reverse_lazy('mld:all')

    def get(self, request, pk=None):
        form = CreateGraphForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateGraphForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        graph = form.save(commit=False)
        graph.owner = self.request.user

        graph.save()
        return redirect(self.success_url)

class DatasetCreateView(LoginRequiredMixin, View):
    template_name = 'mld/dataset_form.html'
    success_url = reverse_lazy('mld:all_datasets')

    def get(self, request, pk=None):
        form = CreateDatasetForm()
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        form = CreateDatasetForm(request.POST, request.FILES or None)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        # Add owner to the model before saving
        dataset = form.save(commit=False)
        dataset.owner = self.request.user

        dataset = load_dataset(dataset)

        dataset.save()

        return redirect(self.success_url)

def return_dataframe(dataset):
    if dataset.data == None and dataset.title != 'Wisconsin Breast Cancer Dataset' and dataset.title != 'Boston Housing Dataset':
        df = pd.DataFrame()
        return df
    else:
        if dataset.title == 'Wisconsin Breast Cancer Dataset':
            cancer = load_breast_cancer()
            df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))
        elif dataset.title == 'Boston Housing Dataset':
            boston_data, boston_target=mglearn.datasets.load_extended_boston()
            df = pd.DataFrame(np.c_[boston_data, boston_target])
            # boston = load_boston()
            # df = pd.DataFrame(boston.data, columns=boston.feature_names)
            # df['MEDV'] = boston.target
        elif dataset.file_type.title == 'CSV':
            # data_encoded = dataset.data
            # data_decoded = data_encoded.decode()
            data_decoded = dataset.data
            if dataset.has_header == True:
                #df = pd.read_csv(StringIO(data_decoded), header=0)
                df = pd.read_csv(BytesIO(data_decoded), header=0)
            else:
                # df = pd.read_csv(StringIO(data_decoded), header=None)
                df = pd.read_csv(BytesIO(data_decoded), header=None)
        elif dataset.file_type.title == 'Excel .xlsx':
            if dataset.has_header == True:
                df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=0)
            else:
                df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=None)
        elif dataset.file_type.title == 'Excel .xls':
            if dataset.has_header == True:
                df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=0)
            else:
                df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=None)
    return df

def load_dataset(dataset):
    if dataset.data == None and dataset.title != 'Wisconsin Breast Cancer Dataset' and dataset.title != 'Boston Housing Dataset':
        return dataset
    else:
        if dataset.title == 'Wisconsin Breast Cancer Dataset':
            cancer = load_breast_cancer()
            df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns= np.append(cancer['feature_names'], ['target']))
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
        elif dataset.title == 'Boston Housing Dataset':
            boston_data, boston_target=mglearn.datasets.load_extended_boston()
            df = pd.DataFrame(np.c_[boston_data, boston_target])
            # boston = load_boston()
            # df = pd.DataFrame(boston.data, columns=boston.feature_names)
            # df['MEDV'] = boston.target
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
        elif dataset.file_type.title == 'CSV':
            # data_encoded = dataset.data
            # data_decoded = data_encoded.decode()
            data_decoded = dataset.data
            if dataset.has_header == True:
                # df = pd.read_csv(StringIO(data_decoded), header=0)
                df = pd.read_csv(BytesIO(data_decoded), header=0)
            else:
                #df = pd.read_csv(StringIO(data_decoded), header=None)
                df = pd.read_csv(BytesIO(data_decoded), header=None)
            # df = pd.read_csv(StringIO(data_decoded), header=None)
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
        elif dataset.file_type.title == 'Excel .xlsx':
            if dataset.has_header == True:
                df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=0)
            else:
                df = pd.read_excel(BytesIO(dataset.data), engine='openpyxl', skiprows=[0], header=None)
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
            df_target=df_target.astype('str')
        elif dataset.file_type.title == 'Excel .xls':
            if dataset.has_header == True:
                df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=0)
            else:
                df = pd.read_excel(BytesIO(dataset.data), skiprows=[0], header=None)
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
            df_target=df_target.astype('str')

        dataset.dimensionality = len(df_data.columns)
        dataset.samples = len(df)
        s = df_target.to_numpy()
        dataset.target_data = s.tobytes()
        if dataset.ml_type.id == 1:
            vcounts = df_target.value_counts()
            dataset.number_of_classes = len(vcounts)
            dataset.class_1_count = vcounts.iloc[0]
            dataset.class_2_count = vcounts.iloc[1]
            dataset.class_1_label = str(vcounts.index.values[0])
            dataset.class_2_label = str(vcounts.index.values[1])
        else:
            dataset.target_mean = '{:0.2f}'.format(df_target.mean())
            dataset.target_std = '{:0.2f}'.format(df_target.std())
            dataset.target_min = '{:0.2f}'.format(df_target.min())
            dataset.target_max = '{:0.2f}'.format(df_target.max())
    return dataset

class MldUpdateView(LoginRequiredMixin, View):
    template_name = 'mld/mld_form.html'
    success_url = reverse_lazy('mld:all')

    def get(self, request, pk):
        entry = get_object_or_404(Entry, id=pk, owner=self.request.user)
        form = CreateForm(instance=entry)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        entry = get_object_or_404(Entry, id=pk, owner=self.request.user)
        form = CreateForm(request.POST, request.FILES or None, instance=entry)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        entry = form.save(commit=False)
        entry = ml_train_test(entry)
        entry.save()

        return redirect(self.success_url)

class GraphUpdateView(LoginRequiredMixin, View):
    template_name = 'mld/graph_form.html'
    success_url = reverse_lazy('mld:all_graphs')

    def get(self, request, pk):
        graph = get_object_or_404(Graph, id=pk, owner=self.request.user)
        form = CreateGraphForm(instance=graph)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        graph = get_object_or_404(Graph, id=pk, owner=self.request.user)
        form = CreateGraphForm(request.POST, request.FILES or None, instance=graph)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        graph = form.save(commit=False)


        graph.save()

        return redirect(self.success_url)

class DatasetUpdateView(LoginRequiredMixin, View):
    template_name = 'mld/dataset_form.html'
    success_url = reverse_lazy('mld:all_datasets')

    def get(self, request, pk):
        dataset = get_object_or_404(Dataset, id=pk, owner=self.request.user)
        form = CreateDatasetForm(instance=dataset)
        ctx = {'form': form}
        return render(request, self.template_name, ctx)

    def post(self, request, pk=None):
        dataset = get_object_or_404(Dataset, id=pk, owner=self.request.user)
        form = CreateDatasetForm(request.POST, request.FILES or None, instance=dataset)

        if not form.is_valid():
            ctx = {'form': form}
            return render(request, self.template_name, ctx)

        dataset = form.save(commit=False)

        dataset = load_dataset(dataset)

        dataset.save()

        return redirect(self.success_url)


class MldDeleteView(OwnerDeleteView):
    model = Entry

class GraphDeleteView(OwnerDeleteView):
    model = Graph

class DatasetDeleteView(OwnerDeleteView):
    model = Dataset

def ml_train_test(obj):

    start = time.time()

    if obj.dataset.title == 'Wisconsin Breast Cancer Dataset':
        X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
    elif obj.dataset.title == 'Boston Housing Dataset':
        boston_data, boston_target=mglearn.datasets.load_extended_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston_data, boston_target, random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
    # elif obj.dataset.index == 1:
    #     X_train, X_test, y_train, y_test = train_test_split(bank_data, bank_target, stratify=bank_target, random_state=66)
    # elif obj.dataset.index == 2:
    #     X_train, X_test, y_train, y_test = train_test_split(survival_data, survival_target, stratify=survival_target, random_state=66)
    # elif obj.dataset.index == 3:
    #     X_train, X_test, y_train, y_test = train_test_split(sonar_data, sonar_target, stratify=sonar_target, random_state=66)
    # elif obj.dataset.index == 4:
    #     X_train, X_test, y_train, y_test = train_test_split(ionosphere_data, ionosphere_target, stratify=ionosphere_target, random_state=66)
    # elif obj.dataset.index == 5:
    #     X_train, X_test, y_train, y_test = train_test_split(pima_data, pima_target, stratify=pima_target, random_state=66)
    # elif obj.dataset.index == 6:
    else:
        if obj.dataset.file_type.title == 'CSV':
            # data_encoded = obj.dataset.data
            # data_decoded = data_encoded.decode()
            data_decoded = obj.dataset.data
            if obj.dataset.has_header == True:
                #df = pd.read_csv(StringIO(data_decoded), header=0)
                df = pd.read_csv(BytesIO(data_decoded), header=0)
            else:
                #df = pd.read_csv(StringIO(data_decoded), header=None)
                df = pd.read_csv(BytesIO(data_decoded), header=None)
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
        elif obj.dataset.file_type.title == 'Excel .xlsx':
            if obj.dataset.has_header == True:
                df = pd.read_excel(BytesIO(obj.dataset.data), engine='openpyxl', skiprows=[0], header=0)
            else:
                df = pd.read_excel(BytesIO(obj.dataset.data), engine='openpyxl', skiprows=[0], header=None)
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
            # df_target=df_target.astype('str')
        elif obj.dataset.file_type.title == 'Excel .xls':
            if obj.dataset.has_header == True:
                df = pd.read_excel(BytesIO(obj.dataset.data), skiprows=[0], header=0)
            else:
                df = pd.read_excel(BytesIO(obj.dataset.data), skiprows=[0], header=None)
            df_data = df.iloc[:,0:-1]
            df_target = df.iloc[:, -1]
            # df_target=df_target.astype('str')
        if obj.dataset.ml_type.id == 1:
            X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, stratify=df_target, random_state=66)
        elif obj.dataset.ml_type.id == 2:
            X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, random_state=66)

    model = obj.algorithm.index

    param1 = obj.parameter_1_value
    try:
        param1 = float(param1)
    except:
        param1 = 1

    param2 = obj.parameter_2_value
    try:
        param2 = float(param2)
    except:
        param2 = 1

    param3 = obj.parameter_3_value
    try:
        param3 = float(param3)
    except:
        param3 = 1

    if model == 0:
        clf = KNeighborsClassifier(n_neighbors=int(param1)).fit(X_train, y_train)
    elif model == 1:
        clf = LogisticRegression(C=param1).fit(X_train, y_train)
    elif model == 2:
        clf = LinearSVC(C=param1).fit(X_train, y_train)
    elif model == 3:
        clf = DecisionTreeClassifier(max_depth = int(param1), random_state=0).fit(X_train, y_train)
    elif model == 4:
        clf = RandomForestClassifier(n_estimators = int(param1), random_state=0).fit(X_train, y_train)
    elif model == 5:
        clf = GradientBoostingClassifier(max_depth = int(param1), random_state=0).fit(X_train, y_train)
    elif model == 6:
        clf = MLPClassifier(solver='lbfgs',  hidden_layer_sizes=[int(param1)], max_iter=int(param2), alpha=param3, random_state=0).fit(X_train, y_train)
    elif model == 7:
        clf = GaussianNB(var_smoothing = param1).fit(X_train, y_train)
    elif model == 8:
        clf = MultinomialNB(alpha = param1).fit(X_train, y_train)
    elif model == 9:
        reg = KNeighborsRegressor(n_neighbors=int(param1)).fit(X_train, y_train)
    elif model == 10:
        reg = LinearRegression().fit(X_train, y_train)
    elif model == 11:
        reg = Ridge(alpha=param1).fit(X_train, y_train)
    elif model == 12:
        reg = Lasso(alpha=param1).fit(X_train, y_train)


    end = time.time()

    modeltime = end - start
    modeltime = '{:0.3f}'.format(modeltime)
    obj.time = modeltime

    if model < 9:
        train_accuracy = clf.score(X_train, y_train)
        train_accuracy = '{:0.3f}'.format(train_accuracy)
        y_pred = clf.predict(X_test)
        precision = precision_score(y_test, y_pred, average="macro")
        precision = '{:0.3f}'.format(precision)
        f1 = f1_score(y_test, y_pred, average="macro")
        f1 = '{:0.3f}'.format(f1)
        recall = recall_score(y_test, y_pred, average="macro")
        recall = '{:0.3f}'.format(recall)
        test_accuracy = clf.score(X_test, y_test)
        test_accuracy = '{:0.3f}'.format(test_accuracy)

        obj.precision = precision
        obj.recall = recall
        obj.f1 = f1
        obj.train_accuracy = train_accuracy
        obj.test_accuracy = test_accuracy

        confusion = confusion_matrix(y_test, y_pred)
        obj.TN  = confusion[0,0]
        obj.FP = confusion[0,1]
        obj.FN = confusion[1,0]
        obj.TP = confusion[1,1]

        if model == 1 or model == 2:
            # if model is Logistic Regression or LinearSVC use decision_function to calculate ROC curve:
            fpr, tpr, thresholds = roc_curve(y_test, clf.decision_function(X_test))
            precision_c, recall_c, thresholds2 = precision_recall_curve(y_test, clf.decision_function(X_test))
            avg_precision = average_precision_score(y_test, clf.decision_function(X_test))
        else:
            # otherwise use predict_proba to calculate ROC curve:
            y_scores = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
            precision_c, recall_c, thresholds2 = precision_recall_curve(y_test, clf.predict_proba(X_test)[:,1])
            avg_precision = average_precision_score(y_test, clf.predict_proba(X_test)[:,1])

        avg_precision = '{:0.3f}'.format(avg_precision)
        obj.avg_precision = avg_precision

        roc_auc = auc(fpr, tpr)
        roc_auc = '{:0.3f}'.format(roc_auc)
        obj.roc_auc = roc_auc

        # save numpy arrays for ROC curve to binary fields of entry model
        obj.fpr = fpr.tobytes()
        obj.tpr = tpr.tobytes()
        obj.thresholds = thresholds.tobytes()

        obj.precision_c = precision_c.tobytes()
        obj.recall_c = recall_c.tobytes()
        obj.thresholds2 = thresholds2.tobytes()
        df_new = return_dataframe(obj.dataset)
        s = df_new.iloc[:, -1]
        target_data = s.to_numpy()
        obj.target_data = target_data.tobytes()
    else:
        y_train_predict = reg.predict(X_train)
        y_test_predict = reg.predict(X_test)

        rmse_train_score = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
        rmse_train_score = '{:0.3f}'.format(rmse_train_score)
        rmse_test_score = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
        rmse_test_score = '{:0.3f}'.format(rmse_test_score)

        # r2_train_score = reg.score(X_train, y_train)
        r2_train_score = r2_score(y_train, y_train_predict)

        r2_train_score = '{:0.3f}'.format(r2_train_score)

        # r2_test_score = reg.score(X_test, y_test)
        r2_test_score = r2_score(y_test, y_test_predict)

        r2_test_score = '{:0.3f}'.format(r2_test_score)

        obj.r2_train_score = r2_train_score
        obj.r2_test_score = r2_test_score
        obj.rmse_train = rmse_train_score
        obj.rmse_test = rmse_test_score

        df_new = return_dataframe(obj.dataset)
        s = df_new.iloc[:, -1]
        target_data = s.to_numpy()
        obj.target_data = target_data.tobytes()

    return obj