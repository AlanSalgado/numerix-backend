from django.urls import path
from .views import PredictDigit

urlpatterns = [
    path("predict/", PredictDigit.as_view(), name="predict")
]