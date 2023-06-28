using Distributed
@everywhere Distributed
@distributed for i=1:100
    print(i)
    print("-")
end
