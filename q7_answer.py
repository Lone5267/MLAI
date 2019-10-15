# Question 7 Code Answer

q = 2 # the dimension of our map of the 'library'
learn_rate = 0.1
U = pd.DataFrame(np.random.normal(size=(nUsersInExample, q))*0.001, index=my_batch_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.001, index=indexes_unique_movies)

def stochastic_objective_gradient(Y, U, V):
    rand_row = np.random.randint(Y.shape[0])
    row = Y.iloc[rand_row] # randomly select a row from Y
    user = row['users']
    film = row['movies']
    rating = row['ratings']
    prediction = np.dot(U.loc[user], V.loc[film])
    diff = prediction - rating
    obj = diff*diff
    gU = 2*diff*V.loc[film]
    gV = 2*diff*U.loc[user]
    return obj, gU, gV

iterations = 100000
for i in range(iterations):
    obj, gU, gV = stochastic_objective_gradient(Y, U, V)
    if i%10000 == 0:
        overall_obj = 0
        for j in range(Y.shape[0]):
            row = Y.iloc[j]
            rating = row['ratings']
            user = row['users']
            film = row['movies']
            prediction = np.dot(U.loc[user], V.loc[film])
            diff = prediction - rating
            overall_obj += diff*diff
        print("Iteration", i, "Objective function: ", overall_obj)
    U -= learn_rate*gU
    V -= learn_rate*gV  
