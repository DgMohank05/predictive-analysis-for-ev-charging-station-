# ev_project/urls.py
from django.contrib import admin
from django.urls import path, include
from ev_app.views import fetch_stations  # Import your view for testing

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ev/', include('ev_app.urls')),  # Include app URLs
    path('', fetch_stations, name='home'),  # Add this line to handle the root path
]
