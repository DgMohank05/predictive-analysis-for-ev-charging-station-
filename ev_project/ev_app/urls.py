# ev_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('fetch_stations/', views.fetch_stations, name='fetch_stations'),
    path("predict/", views.predict_usage, name="predict_usage"),
]
