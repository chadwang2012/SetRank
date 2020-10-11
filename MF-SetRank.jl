
# This is the implementation for MF-SetRank.
# This code is based on the implementation of SQL-Rank on https://github.com/wuliwei9278/SQL-Rank.

function logit(x)
    return 1.0/(1+exp(-x))
end

function comp_m(rows, cols, U, V)
    m = zeros(length(rows));
    for i = 1:length(rows)
        m[i] = logit(dot(U[:,cols[i]], V[:,rows[i]]));
    end
    return m
end

function objective(index, m, rows, d1, lambda, U, V)
    res = 0.0;
    for i = 1:d1
        tt = 0.0;
        d_bar = index[i+1] - index[i];
        for t = d_bar:-1:1
            tmp = m[index[i] - 1 + t];
            tt += exp(m[index[i] - 1 + t]);
            res -= tmp;
            res += log(tt);
        end
    end
    res += lambda / 2 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2);
    return res
end

function comp_gradient_ui(rows, cols, index, d_bar, m, i, V, r, ratio)
    nowOnes = div(d_bar, 1 + ratio);
    cc = zeros(d_bar);

    total = 0
    for t = (nowOnes+1):d_bar
        ttt = m[index[i] - 1 + t];
        total += exp(ttt);
    end
    total_i = zeros(nowOnes);
    total_sum = 0
    for sel_i = 1:nowOnes
        ttt = m[index[i] - 1 + sel_i];
        cc[sel_i] -= ttt * (1 - ttt);
        total_i[sel_i] = exp(ttt) + total;
        total_sum += 1 / total_i[sel_i]
    end
    for t = (nowOnes+1):d_bar
        ttt = m[index[i] - 1 + t];
        cc[t] += exp(ttt) * ttt * (1 - ttt) * total_sum;
    end
    for sel_i = 1:nowOnes
        ttt = m[index[i] - 1 + sel_i];
        cc[sel_i] += exp(ttt) * ttt * (1 - ttt) / total_i[sel_i];
    end

    res = zeros(r);
    for t = 1:d_bar
        res += cc[t] * V[:,rows[index[i] - 1 + t]];
    end
    return res
end

function comp_gradient_U(rows, cols, index, m, U, V, s, d1, r, lambda, ratio)
    grad_U = zeros(size(U));
    for i = 1:d1
        d_bar = index[i+1] - index[i];
        grad_U[:,i] = comp_gradient_ui(rows, cols, index, d_bar, m, i, V, r, ratio);
    end
    grad_U += lambda * U;
    return grad_U
end

function obtain_U(rows, cols, index, U, V, s, d1, r, lambda, ratio)
    m = comp_m(rows, cols, U, V);
    grad_U = comp_gradient_U(rows, cols, index, m, U, V, s, d1, r, lambda, ratio);
    U = U - s * grad_U;
    m = comp_m(rows, cols, U, V);
    return U, m
end

function comp_gradient_V(rows, cols, index, m, U, V, s, d1, r, lambda, ratio)
    grad_V = zeros(size(V));
    for i = 1:d1
        d_bar = index[i+1] - index[i];
        cc = zeros(d_bar);
        nowOnes = div(d_bar, 1 + ratio);

        total = 0
        for t = (nowOnes+1):d_bar
            ttt = m[index[i] - 1 + t];
            total += exp(ttt);
        end
        total_i = zeros(nowOnes);
        total_sum = 0
        for sel_i = 1:nowOnes
            ttt = m[index[i] - 1 + sel_i];
            cc[sel_i] -= ttt * (1 - ttt);
            total_i[sel_i] = exp(ttt) + total;
            total_sum += 1 / total_i[sel_i]
        end
        for t = (nowOnes+1):d_bar
            ttt = m[index[i] - 1 + t];
            cc[t] += exp(ttt) * ttt * (1 - ttt) * total_sum;
        end
        for sel_i = 1:nowOnes
            ttt = m[index[i] - 1 + sel_i];
            cc[sel_i] += exp(ttt) * ttt * (1 - ttt) / total_i[sel_i];
        end

        for t = 1:d_bar
            j = rows[index[i] - 1 + t]
            grad_V[:,j] += cc[t] * U[:,i]
#             println("t: ", t, "cc[t]: ", cc[t]);
#             println("grad_V[:,j]: ", grad_V[:,j]);
        end
    end
    grad_V += lambda * V;
    return grad_V
end

function obtain_V(rows, cols, index, m, U, V, s, d1, r, lambda, ratio)
    grad_V = comp_gradient_V(rows, cols, index, m, U, V, s, d1, r, lambda, ratio);
    V = V - s * grad_V;
    return V
end

function stochasticQueuing(rows, index, d1, d2, ratio)
    new_rows = zeros(Int, size(rows)[1]);
    for i = 1:d1
        nowlen = index[i + 1] - index[i];
        nowOnes = div(nowlen, 1 + ratio);
        newOrder = shuffle(1:nowOnes);
        rows_set = Set{Int}();
        for j = 1:nowOnes
            oldIdx = index[i] + j - 1;
            row_j = rows[oldIdx];
            push!(rows_set, row_j);
            newIdx = index[i] + newOrder[j] - 1;
            new_rows[newIdx] = row_j;
        end
        nowStart = index[i] + nowOnes;
        nowEnd = index[i + 1] - 1;
        for j = nowStart:nowEnd
            while true
                row_idx = rand(1:d2);
                if !(row_idx in rows_set)
                    new_rows[j] = row_idx;
                    push!(rows_set, row_idx);
                    break;
                end
            end
        end
    end
    return new_rows
end

function evaluate(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t, testsize, K)
    precision = zeros(length(K));
    recall = zeros(length(K));
    map = zeros(length(K));
    score = V'* U;
    for i = 1:d1
        precision_tmp = zeros(length(K));
        recall_tmp = zeros(length(K));
        map_tmp = zeros(length(K));
        tmp = nzrange(Y, i);
        test = Set{Int64}();
        for j in tmp
            push!(test, rows_t[j]);
        end
        if isempty(test)
            continue
        end
        tmp = nzrange(X, i);
        vals_d2_bar = vals[tmp];
        train = rows[tmp];
        score[train, i] = -999;
        p = sortperm(score[:, i], rev = true);
        cc = 0;
        for c = 1: K[length(K)]
            j = p[c];
            if j in test
                cc += 1;
                for k in length(K):-1:1
                    if c <= K[k]
                        precision_tmp[k] += 1;
                        recall_tmp[k] += 1;
                        map_tmp[k] += cc / c;
                    else
                        break;
                    end
                end
            end
        end
        ntest = length(test);
        precision += precision_tmp ./ K;
        recall += recall_tmp / ntest;
        map += map_tmp ./ K;
    end
    return precision/testsize, recall/testsize, map/testsize
end


train = "data/ml1m_oc_50_train_ratings.csv"
test = "data/ml1m_oc_50_test_ratings.csv"

T = 1;
ratio = 2;
learning_rate = 0.3;
decay_rate = 0.97;
lambda = 1.4;
r = 200;

filename1 = string("dumper/", "mfsetrank", ".csv")

X = readdlm(train, ',' , Int64);
x = vec(X[:,1]);
y = vec(X[:,2]);
v = vec(X[:,3]);
Y = readdlm(test, ',' , Int64);
xx = vec(Y[:,1]);
yy = vec(Y[:,2]);
vv = vec(Y[:,3]);
n = max(maximum(x), maximum(xx));
msize = max(maximum(y), maximum(yy));
testsize = length(unique(xx));

X = sparse(x, y, v, n, msize); # userid by movieid
Y = sparse(xx, yy, vv, n, msize);
X = X';
Y = Y';
rows = rowvals(X);
vals = nonzeros(X);
cols = zeros(Int, size(vals)[1]);
index = zeros(Int, n + 1);

d2, d1 = size(X);
cc = 0;
new_len = 0;
new_index = zeros(Int, d1 + 1);
new_index[1] = 1;
for i = 1:d1
    index[i] = cc + 1;
    tmp = nzrange(X, i);
    nowlen = size(tmp)[1];
    newlen = nowlen * (1 + ratio);
    new_len += newlen;
    new_index[i + 1] = new_index[i] + newlen;
    for j = 1:nowlen
        cc += 1;
        cols[cc] = i;
    end
end
index[d1 + 1] = cc + 1;

new_rows = zeros(Int, new_len);
new_cols = zeros(Int, new_len);
new_vals = zeros(Int, new_len);
for i = 1:d1
    rows_set = Set{Int}();
    for j = index[i]:(index[i + 1] - 1)
        push!(rows_set, rows[j]);
    end
    nowlen = new_index[i + 1] - new_index[i];
    nowOnes = div(nowlen, 1 + ratio);
    for j = 1:nowOnes
        new_rows[new_index[i] + j - 1] = rows[index[i] + j - 1];
        new_cols[new_index[i] + j - 1] = i;
        new_vals[new_index[i] + j - 1] = vals[index[i] + j - 1];
    end
    nowStart = new_index[i] + nowOnes;
    nowEnd = new_index[i + 1] - 1;
    for j = nowStart:nowEnd
        while true
            row_idx = rand(1:d2);
            if !(row_idx in rows_set)
                new_rows[j] = row_idx;
                new_cols[j] = i;
                new_vals[j] = 0.0;
                push!(rows_set, row_idx);
                break;
            end
        end
    end
end

rows_t = rowvals(Y);
vals_t = nonzeros(Y);
cols_t = zeros(Int, size(vals_t)[1]);
index_t =  zeros(Int, n + 1)
cc = 0;
for i = 1:d1
    index_t[i] = cc + 1;
    tmp = nzrange(Y, i);
    nowlen = size(tmp)[1];
    for j = 1:nowlen
        cc += 1
        cols_t[cc] = i
    end
end
index_t[d1 + 1] = cc + 1;

srand(123456789);
U = 0.1*randn(r, d1);
V = 0.1*randn(r, d2);
m = comp_m(new_rows, new_cols, U, V);
println("rank: ", r, ", ratio of 0 vs 1: ", ratio, ", lambda:", lambda, ", learning_rate: ", learning_rate);

num_epoch = 100;
num_iterations_per_epoch = 1;
K = [1, 5, 10, 20];
precision = zeros(Float64, num_epoch, 4)
recall = zeros(Float64, num_epoch, 4)
map = zeros(Float64, num_epoch, 4)
println("iter time objective_function precision@K = 1, 5, 10");
obj = objective(new_index, m, new_rows, d1, lambda, U, V);
p1,p2,p3=evaluate(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t, testsize, K);
println("[", 0, ",", obj, ", ", p1," ",p2," ",p3, "],");

totaltime = 0.00000;
nowobj = obj;
for epoch = 1:num_epoch
    tic();
    for iter = 1:num_iterations_per_epoch
        U, m = obtain_U(new_rows, new_cols, new_index, U, V, learning_rate, d1, r, lambda, ratio);
        V = obtain_V(new_rows, new_cols, new_index, m, U, V, learning_rate, d1, r, lambda, ratio);
    end

    new_rows = stochasticQueuing(new_rows, new_index, d1, d2, ratio);

    totaltime += toq();
    if (epoch - 1) % T == 0
        learning_rate = learning_rate * decay_rate
        p1,p2,p3=evaluate(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t, testsize, K);
        precision[epoch, :] = p1
        recall[epoch, :] = p2
        map[epoch, :] = p3
        m = comp_m(new_rows, new_cols, U, V);
        nowobj = objective(new_index, m, new_rows, d1, lambda, U, V);
        println("[", epoch, ", ", totaltime, ", ", nowobj, ", ", p1,", ",p2,", ",p3, "],");
    else
        m = comp_m(new_rows, new_cols, U, V);
        nowobj = objective(new_index, m, new_rows, d1, lambda, U, V);
        println("[", epoch, ", ", totaltime, ", ", nowobj);
    end
end

writedlm(filename1, [precision recall map])
