def precizie_machine():
   u = 1.0
   m = 0
   while 1.0 + (u/10) != 1.0:
        m = m + 1
        u/=10
   return m

print("Value of me is:", precizie_machine())