#!/bin/bash
echo "=== WHAT IS ON PORT 8000 ==="
ss -tlnp 2>/dev/null | grep 8000 || netstat -tlnp 2>/dev/null | grep 8000 || echo "Cannot check ports"
echo "=== WHAT IS ON PORT 8080 ==="
ss -tlnp 2>/dev/null | grep 8080 || netstat -tlnp 2>/dev/null | grep 8080 || echo "Nothing on 8080"
echo "=== WHAT IS ON PORT 3000 ==="
ss -tlnp 2>/dev/null | grep 3000 || netstat -tlnp 2>/dev/null | grep 3000 || echo "Nothing on 3000"
