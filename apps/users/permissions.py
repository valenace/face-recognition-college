from functools import wraps
from django.core.exceptions import PermissionDenied
from django.contrib.auth.mixins import AccessMixin
from django.shortcuts import redirect

def role_required(*roles):
    """
    decorador para vistas basadas en funciones que restringe el acceso
    a usuarios que tengan uno de los roles especificados.
    """
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            
            # los desarrolladores siempre tienen acceso completo
            if request.user.is_superuser or request.user.role in roles:
                return view_func(request, *args, **kwargs)
                
            raise PermissionDenied
        return _wrapped_view
    return decorator


class RoleRequiredMixin(AccessMixin):
    """
    mixin para vistas basadas en clases que restringe el acceso
    a usuarios que tengan uno de los roles especificados.
    """
    allowed_roles = []

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return self.handle_no_permission()
        
        # los desarrolladores tienen acceso completo
        if request.user.is_superuser or request.user.role in self.allowed_roles:
            return super().dispatch(request, *args, **kwargs)
            
        raise PermissionDenied
