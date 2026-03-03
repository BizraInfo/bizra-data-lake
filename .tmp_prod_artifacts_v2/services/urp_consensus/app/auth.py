try:
    from _shared.app.auth import require_admin
except ImportError:  # Backward compatibility for non-container execution layouts.
    from services._shared.app.auth import require_admin
