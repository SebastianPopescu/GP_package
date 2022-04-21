#############################################################
###### KL-div between Multivariate Normal distributions #####
#############################################################

def KL_inverse_free(q_mu, q_var_choleksy,  Sigma_mm_inverse, type_var, white,  L_K_A):

    ### Kl-Div between posterior and prior over inducing point values
    ### q_mu -- shape (num_inducing, dim_output)
    ### q_var_cholesky -- shape (dim_output, num_inducing, num_inducing)
    ### posterior_Kmm_inverse -- shape (num_inducing, num_inducing)

    ### TODO -- the L_K_A from propagate_layers has to be changed #####

    if not white:

        S = tf.matmul(q_var_choleksy, q_var_choleksy, transpose_b = True) ### shape -- (dim_output, num_inducing, num_inducing)
        Sigma_mm_inverse = tf.tile(tf.expand_dims(Sigma_mm_inverse, axis = 0), [tf.shape(q_mu)[1],1,1]) ### shape -- (dim_output, num_inducing, num_inducing)
        L_K_A = tf.tile(tf.expand_dims(L_K_A, axis = 0), [tf.shape(q_mu)[1],1,1]) ### shape -- (dim_output, num_inducing, num_inducing)

        kl_term = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_var_choleksy))) 
        kl_term -=  2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_K_A))) 
        kl_term -= tf.cast(tf.shape(q_mu)[0],DTYPE ) * tf.cast(tf.shape(q_mu)[1],DTYPE )
        ### Explicit calculation of trace term 
        ### TODO -- implement Hutchinson trace estimator ### 	
        kl_term += tf.reduce_sum(tf.linalg.trace(tf.matmul(Sigma_mm_inverse, S)))
        q_mu = tf.expand_dims(tf.transpose(q_mu),axis=-1) ### shape (dim_output, num_inducing, 1)
        kl_term += tf.reduce_sum(tf.matmul(tf.matmul(q_mu, Sigma_mm_inverse, transpose_a = True),q_mu))
    
    elif white:

        S = tf.matmul(q_var_choleksy, q_var_choleksy, transpose_b = True) ### shape -- (dim_output, num_inducing, num_inducing)

        kl_term = - 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_var_choleksy))) 
        
        kl_term -= tf.cast(tf.shape(q_mu)[0],DTYPE ) * tf.cast(tf.shape(q_mu)[1],DTYPE ) 	
        ### Explicit calculation of trace term 
        ### TODO -- implement Hutchinson trace estimator ### 	
        kl_term += tf.reduce_sum(tf.linalg.trace(S))
        q_mu = tf.expand_dims(tf.transpose(q_mu),axis=-1) ### shape (dim_output, num_inducing, 1)
        kl_term += tf.reduce_sum(tf.matmul(q_mu, q_mu, transpose_a = True))

    return 0.5 * kl_term
