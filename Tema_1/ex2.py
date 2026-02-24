def precizie_machine():
   u = 1.0
   m = 0
   while 1.0 + (u/10) != 1.0:
        m = m + 1
        u/=10
   return 10**(-m)

print("Neasociativitatea addition:")

x = 1.0
y = precizie_machine()/10.0
z = precizie_machine()/10.0
print("Operation of addition is associative." if(((x+y) + z ) == (x + (y + z))) else "Operation of addition is not associative")

print("Neasociativitatea multiplication:")

x = 1e308
y = 10.0
z = 0.1
print("Operation of multiplication is associative." if(((x*y) * z ) == (x * (y * z))) else "Operation of multiplication is not associative")



