function mu=trapf1(x,A)
a=A(1); b=A(2); c=A(3); d=A(4);
if (x > a & x <=b)
    mu=(x-a)/(b-a);
elseif (x>b & x <=c)
    mu=1;
elseif (x>c & x <=d)
    mu=(d-x)/(d-c);
else
    mu=0;
end