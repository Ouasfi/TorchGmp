#include <torch/extension.h>
#include <vector>
#include <libhcs.h>
#include <gmp.h>  
// Keys 
std::vector<std::vector<unsigned long int>> pcs_keys(int modsize=12) {
    pcs_public_key *pk = pcs_init_public_key();
    pcs_private_key *vk = pcs_init_private_key();
    hcs_random *hr = hcs_init_random();

    // Generate a key pair with modulus of size 2048 bits
    pcs_generate_key_pair(pk, vk, hr, modsize);
    //gmp_printf("%Zd\n", mpz_get_ui (hr->rstate->_mp_seed));
    //int(mpz_get_ui (hr->rstate->_mp_seed))
    std::vector<unsigned long int> public_k = {static_cast< unsigned long int>(mpz_get_ui (pk->n)), 
                                               static_cast< unsigned long int>(mpz_get_ui (pk->g)), 
                                               static_cast< unsigned long int>(mpz_get_ui (pk->n2))};

    std::vector<unsigned long int> private_k = {static_cast< unsigned long int>(mpz_get_ui(vk->p)),
                                                static_cast< unsigned long int>(mpz_get_ui(vk->p2)),
                                                static_cast< unsigned long int>(mpz_get_ui(vk->q)), 
                                                static_cast< unsigned long int>(mpz_get_ui(vk->q2)), 
                                                static_cast< unsigned long int>(mpz_get_ui(vk->hp)), 
                                                static_cast< unsigned long int>(mpz_get_ui(vk->hq)), 
                                                static_cast< unsigned long int>(mpz_get_ui(vk->mu)), 
                                                static_cast< unsigned long int>(mpz_get_ui(vk->lambda)),
                                                static_cast< unsigned long int>(mpz_get_ui(vk->n)),  
                                                static_cast< unsigned long int>(mpz_get_ui(vk->n2))};
  return {public_k,private_k };
}
pcs_public_key*  public_key_set(std::vector<unsigned long int> public_key) {
    pcs_public_key *pk = pcs_init_public_key();
    mpz_set_ui(pk->n , public_key[0]);
    mpz_set_ui(pk->g , public_key[1]);
    mpz_set_ui(pk->n2 , public_key[2]);
    

  return pk;
}
pcs_private_key*  private_key_set(std::vector<unsigned long int> private_key) {
    pcs_private_key *vk = pcs_init_private_key();
          mpz_set_ui(vk->p, private_key      [0]);
          mpz_set_ui(vk->p2, private_key     [1]);
          mpz_set_ui(vk->q, private_key      [2]);
          mpz_set_ui(vk->q2, private_key     [3]);
          mpz_set_ui(vk->hp, private_key     [4]);
          mpz_set_ui(vk->hq, private_key     [5]);
          mpz_set_ui(vk->mu, private_key     [6]);
          mpz_set_ui(vk->lambda, private_key [7]);
          mpz_set_ui(vk->n, private_key      [8]);
          mpz_set_ui(vk->n2, private_key     [9]);
  return vk;
}

//Encryption
torch::Tensor encrypt(torch::Tensor z,std::vector<unsigned long int> public_key) {
    pcs_public_key *pk = public_key_set(public_key);
    //pcs_private_key *vk = pcs_init_private_key();
    hcs_random *hr = hcs_init_random();

    // Generate a key pair with modulus of size 2048 bits
    //pcs_generate_key_pair(pk, vk, hr, 2048);
    // libhcs works directly with gmp mpz_t types, so initialize some
    mpz_t a, b;
    mpz_inits(a, b, NULL);

    auto type = z.dtype();
    int n = z.size(0);
    auto s = torch::zeros_like(z, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto r = z[elem_idx].item<long>();
          mpz_set_ui(a, r);
          pcs_encrypt(pk, hr, a, a);
          s[elem_idx] = long(mpz_get_ui (a));
          //mpz_inits(a, b, NULL);
    };
    mpz_clears(a, b, NULL);
    pcs_free_public_key(pk);
    hcs_free_random(hr);
  return s;
}

torch::Tensor decrypt(torch::Tensor z_cipher,std::vector<unsigned long int> private_key) {
    pcs_private_key *vk = private_key_set( private_key);

    mpz_t a, b;
    mpz_inits(a, b, NULL);

    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto r = z_cipher[elem_idx].item<long>();
          mpz_set_ui(a, r);
          pcs_decrypt(vk, a, a);
          s[elem_idx] = long(mpz_get_ui (a));
          //mpz_inits(a, b, NULL);
    };
    mpz_clears(a, b, NULL);
    pcs_free_private_key(vk);

  return s;
}

// Utils
void scale(mpz_t b , int q){
    mpz_t q_m;
    mpz_inits(q_m,NULL);
    mpz_set_ui(q_m, q);
    mpz_fdiv_q(b, b, q_m);
    mpz_clears(q_m ,NULL);
}
int gmp_sign(long int x)
{   
    mpz_t a;
    mpz_inits(a, NULL);

    mpz_set_si(a, x);
    gmp_printf("a = %Zd", a);
    return mpz_sgn(a);
}



// Modular_op : op(a,b) mod(public_key->n2)
//
torch::Tensor modular_mul(torch::Tensor z_cipher,unsigned long int other, 
                          unsigned long int module ,
                          int q=1) {
    /*
    Takes E(Tensor), E(scalar) and computs E(Tensor + scalar)
    */                       
    mpz_t a, b, o, mod;
    mpz_inits(a, b,o, mod,NULL);
    mpz_set_ui(o, other);
    mpz_set_ui(mod, module);
    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto r = z_cipher[elem_idx].item<long>();
          mpz_set_ui(a, r);
          mpz_mul (b, a, o);
          mpz_mod (b, b, mod);
          if (q > 1){scale( b , q);}
          s[elem_idx] = long(mpz_get_ui (b));
          mpz_inits(a, b, NULL);
    };
    mpz_clears(a, b,o, mod, NULL);

  return s;
}
unsigned long int modular_sum(torch::Tensor E_x,
                          std::vector<unsigned long int> public_key) {
    /*
    Takes E(Tensor),  and return E(Tensor.sum())
    */ 
    pcs_public_key *pk = public_key_set(public_key);                        
    mpz_t a, b;
    mpz_inits(a, b,NULL);
      auto type = E_x.dtype();
    int n = E_x.size(0);
    mpz_set_ui(b, E_x[0].item<long>());
    #pragma omp parallel for
    for (int elem_idx = 1; elem_idx < n; elem_idx++) {
          auto x_i = E_x[elem_idx].item<long>();
          mpz_set_ui(a, x_i);
          pcs_ee_add(pk , b, b, a);
    //mpz_inits(a, NULL);
    };
    auto s = static_cast< unsigned long int>(mpz_get_ui (b));
    //mpz_clears(a, b, NULL);

  return s;
}
torch::Tensor modular_add(torch::Tensor E_x,torch::Tensor E_y, 
                          std::vector<unsigned long int> public_key) {
    /*
    Takes E(X), E(Y) and computs E(X + Y)
    */ 
    pcs_public_key *pk = public_key_set(public_key);                        
    mpz_t a, b, o;
    mpz_inits(a, b,o,NULL);
      auto type = E_x.dtype();
    int n = E_x.size(0);
    auto s = torch::zeros_like(E_x, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto x_i = E_x[elem_idx].item<long>();
          auto y_i = E_y[elem_idx].item<long>();
          mpz_set_ui(a, x_i);
          mpz_set_ui(o,  y_i);
          pcs_ee_add(pk , b, a, o);
          s[elem_idx] = long(mpz_get_ui (b));
          mpz_inits(a, b, NULL);
    };
    mpz_clears(a, b,o, NULL);

  return s;
}
torch::Tensor modular_sub(torch::Tensor E_x,torch::Tensor E_y, 
                          std::vector<unsigned long int> public_key) {
    /*
    Takes E(X), E(Y) and computs E(X - Y)
    */ 
    pcs_public_key *pk = public_key_set(public_key);                        
    mpz_t a, b, o, sig;
    mpz_inits(a, b,o,sig, NULL);
    mpz_set_si(sig,  -1);
    //mpz_neg(sig, sig);
    auto type = E_x.dtype();
    int n = E_x.size(0);
    auto s = torch::zeros_like(E_x, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto x_i = E_x[elem_idx].item<long>();
          auto y_i = E_y[elem_idx].item<long>();
          mpz_set_ui(a, x_i);
          mpz_set_ui(o,  y_i);
          pcs_ep_mul(pk, o, o, sig);
          pcs_ee_add(pk , b, a, o);
          //gmp_sign(mpz_get_si (b));
          s[elem_idx] = long(mpz_get_si (b));

    };
    mpz_clears(a, b,o, NULL);

  return s;
}
torch::Tensor mul_enc(torch::Tensor z_cipher,
                      torch::Tensor other, 
                      std::vector<unsigned long int> public_key, 
                      std::vector<unsigned long int> private_key ,
                      int q = 1) {
    /*
    Takes E(X), E(Y) and computs E(X *Y ) mod pv->n2.
    */ 
    pcs_private_key *vk = private_key_set( private_key);
    pcs_public_key *pk = public_key_set(public_key);
    hcs_random *hr = hcs_init_random();
    mpz_t z_m,o_m, b, res;
    mpz_inits(z_m,o_m, b, res,NULL);
    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto z_i = z_cipher[elem_idx].item<long>();
          auto o_i = other[elem_idx].item<long>();
          mpz_set_ui(z_m, z_i);
          pcs_decrypt(vk, z_m, z_m);
          mpz_set_ui(o_m, o_i);
          pcs_decrypt(vk, o_m, o_m);
          mpz_mul (b, z_m, o_m);
          if (q > 1){scale( b , q);}
          pcs_encrypt(pk, hr, b, b);
          mpz_mod (b, b, vk->n2);  
          s[elem_idx] = long(mpz_get_ui (b));
          mpz_inits(z_m, o_m,b, NULL);
    };
    mpz_clears(z_m,o_m, b,res, NULL);
    pcs_free_private_key(vk);
    pcs_free_public_key(pk);
  return s;
}
torch::Tensor mul_dec(torch::Tensor z_cipher,
                      torch::Tensor other, 
                      std::vector<unsigned long int> public_key, 
                      std::vector<unsigned long int> private_key ,
                      int q = 1) {
    /*
    Takes E(X), E(Y) and computs X *Y mod pv->n2.
    */ 
    pcs_private_key *vk = private_key_set( private_key);
    pcs_public_key *pk = public_key_set(public_key);
    hcs_random *hr = hcs_init_random();
    mpz_t z_m,o_m, b, res;
    mpz_inits(z_m,o_m, b, res,NULL);
    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto z_i = z_cipher[elem_idx].item<long>();
          auto o_i = other[elem_idx].item<long>();
          mpz_set_ui(z_m, z_i);
          pcs_decrypt(vk, z_m, z_m);
          mpz_set_ui(o_m, o_i);
          pcs_decrypt(vk, o_m, o_m);
          mpz_mul (b, z_m, o_m);
          if (q > 1){scale( b , q);}
          mpz_mod (b, b, vk->n2);  
          //pcs_encrypt(pk, hr, b, b);
          s[elem_idx] = long(mpz_get_ui (b));
          mpz_inits(z_m, o_m,b, NULL);
    };
    mpz_clears(z_m,o_m, b,res, NULL);
    pcs_free_private_key(vk);
    pcs_free_public_key(pk);
  return s;
}
torch::Tensor mul_enc2(torch::Tensor z_cipher,torch::Tensor other, 
                      std::vector<unsigned long int> public_key, 
                      std::vector<unsigned long int> private_key ) {
    /*
    Takes E(X), E(Y) and computs E(X *Y ) mod pv->n2 using pcs_ep_mul
    */ 
    pcs_private_key *vk = private_key_set( private_key);
    pcs_public_key *pk = public_key_set(public_key);
    hcs_random *hr = hcs_init_random();
    mpz_t z_m,o_m, b, res;
    mpz_inits(z_m,o_m, b, res,NULL);
    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto z_i = z_cipher[elem_idx].item<long>();
          auto o_i = other[elem_idx].item<long>();
          mpz_set_ui(z_m, z_i);
          pcs_decrypt(vk, z_m, z_m);
          mpz_set_ui(o_m, o_i);
          //pcs_decrypt(vk, o_m, o_m);
          pcs_ep_mul(pk, b, o_m, z_m);
          s[elem_idx] = long(mpz_get_ui (b));
          mpz_inits(z_m, o_m,b, NULL);
    };
    mpz_clears(z_m,o_m, b,res, NULL);
    pcs_free_private_key(vk);
    pcs_free_public_key(pk);
  return s;
}

torch::Tensor div_enc(torch::Tensor z_cipher,torch::Tensor d_other, unsigned long int r, std::vector<unsigned long int> public_key) {
    
    /*
    Takes E(c), d and r  and computs E(c)*E(-r/d)
    */
    
    
    pcs_public_key *pk = public_key_set(public_key);
    hcs_random *hr = hcs_init_random();

    mpz_t z_m,o_m, r_neg,  b, res;
    mpz_inits(z_m,o_m, b,r_neg,  res,NULL);
    mpz_set_ui(r_neg, r);
    mpz_neg(r_neg, r_neg);
    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto z_i = z_cipher[elem_idx].item<long>();
          auto o_i = d_other[elem_idx].item<long>();
          mpz_set_ui(z_m, z_i);//z
          mpz_set_ui(o_m, o_i);//d
          mpz_fdiv_q(b, r_neg, o_m);// -r/d
          pcs_encrypt(pk, hr, b, b);//E(-r/d)
          pcs_ee_add(pk, res, z_m, b);//E(c-r/d) = E(c)*E(-r/d) = z_m *b
          s[elem_idx] = long(mpz_get_ui (res));
          mpz_inits(z_m, o_m,b, NULL);
    };
    mpz_clears(z_m,o_m, b,res, NULL);
    pcs_free_public_key(pk);
  return s;
}

torch::Tensor retrieve_mul(torch::Tensor z_cipher, 
                          torch::Tensor other_cipher, 
                          torch::Tensor mul_cipher,
                          std::vector<unsigned long int> public_key,
                          unsigned long int r_a,
                          unsigned long int r_b ) {
    /*
    Takes E(A), E(B), E(M) and  r_a, r_b  and computs E(M)*E(A)^(-r_b)*E(B)^(-r_b)*E(-r_a*r_b)
    */
    pcs_public_key *pk = public_key_set(public_key);
    hcs_random *hr = hcs_init_random();
    mpz_t z_m, o_m, mul_m, b, r_a_m, r_b_m, pow_z, pow_o,r;
    mpz_inits(z_m, o_m, b, mul_m, r_a_m, r_b_m,r, pow_z, pow_o, NULL);
    mpz_set_ui(r_a_m, r_a);
    mpz_set_ui(r_b_m, r_b);

    mpz_mul (r, r_a_m, r_b_m);
    mpz_neg(r,r);
    pcs_encrypt(pk, hr, r, r);
    mpz_neg(r_a_m,r_a_m);
    mpz_neg(r_b_m,r_b_m);
    
    auto type = z_cipher.dtype();
    int n = z_cipher.size(0);
    auto s = torch::zeros_like(z_cipher, type);

    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto z_i = z_cipher[elem_idx].item<long>();
          auto o_i = other_cipher[elem_idx].item<long>();
          auto mul_i = mul_cipher[elem_idx].item<long>();
          
          mpz_set_ui(z_m, z_i);
          mpz_set_ui(o_m, o_i);
          mpz_set_ui(mul_m, mul_i);
          
          pcs_ep_mul(pk, pow_z, z_m, r_a_m);
          pcs_ep_mul(pk, pow_o, o_m, r_b_m);
          
          pcs_ee_add(pk, b, mul_m, r);
          pcs_ee_add(pk, b, b, pow_z);
          pcs_ee_add(pk, b, b, pow_o);

          s[elem_idx] = long(mpz_get_ui (b));
          mpz_inits(z_m, o_m,pow_z,pow_o,mul_m,b, NULL);
    };
    mpz_clears(z_m,o_m,mul_m ,b,pow_z, pow_o,NULL);
    pcs_free_public_key(pk);
  return s;
}


torch::Tensor modular_pow(torch::Tensor E_x, 
                          std::vector<unsigned long int> public_key,
                          long int r_a ) {
    /*
    Takes E(x),  r  and computs E(c*r) =E(c)^r.
    */
    
    pcs_public_key *pk = public_key_set(public_key);
    hcs_random *hr = hcs_init_random();
    mpz_t x_m, b, r_a_m, pow_x;
    mpz_inits( x_m, b, r_a_m, pow_x, NULL);
    mpz_set_si(r_a_m, r_a);
    if (r_a < 0){
      mpz_neg(r_a_m,r_a_m);
    }
    auto type = E_x.dtype();
    int n = E_x.size(0);
    auto s = torch::zeros_like(E_x, type);

    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < n; elem_idx++) {
          auto x_i = E_x[elem_idx].item<long>();
          
          mpz_set_si(x_m, x_i);
                    
          pcs_ep_mul(pk, pow_x, x_m, r_a_m);          

          s[elem_idx] = long(mpz_get_si (pow_x));
    };
    mpz_clears(x_m, b, r_a_m, pow_x,NULL);
    pcs_free_public_key(pk);

  return s;
}
int test(int x, int y, int z,  int mod)
{
    // initialize data structures
    pcs_public_key *pk = pcs_init_public_key();
    pcs_private_key *vk = pcs_init_private_key();
    hcs_random *hr = hcs_init_random();
    hcs_random *hr_2 = hcs_init_random();

    // Generate a key pair with modulus of size 2048 bits
    pcs_generate_key_pair(pk, vk, hr, mod);

    // libhcs works directly with gmp mpz_t types, so initialize some
    mpz_t a, b, c, d, e;
    mpz_inits(a, b, c,d, e, NULL);
    //set a = x, b = r 
    mpz_set_si(a, x);
    mpz_set_si(b, y);
    gmp_printf("a = %Zd\nr = %Zd\n", a, b);
    gmp_printf("v = %Zd\nv = %Zd\n", vk->n2, vk->n);
    mpz_set_si(e, mpz_sgn(b)); 
    gmp_printf("e = %Zd\n", e ); // can use all gmp functions still
    //encrypt a , r
    pcs_encrypt(pk, hr_2, a, a);
    pcs_encrypt(pk, hr_2, b, b);  // Encrypt b (= 76) and store back into b
    gmp_printf("a = %Zd\nr = %Zd\n", a, b); // can use all gmp functions still
    
    
    pcs_decrypt(vk, a,a);  // Encrypt b (= 76) and store back into b
    pcs_decrypt(vk, b, b);  // Encrypt b (= 76) and store back into b
    gmp_printf("a = %Zd\nr = %Zd\n", a, b); // can use all gmp functions still
    mpz_set_si(e, mpz_sgn(b)); 
    gmp_printf("e = %Zd\n", e ); // can use all gmp functions still
    
    pcs_encrypt(pk, hr_2, a, a);  // Encrypt b (= 76) and store back into b
    pcs_encrypt(pk, hr_2, b, b);  // Encrypt b (= 76) and store back into b
    // multiplication of encrypted a , r
    // d = E(a)*E(r)
    mpz_mul (d, a, b);
    //decrypt d 
    pcs_decrypt(vk, d, d); 
    gmp_printf("z = a+r = %Zd\n", d);
    // e = z
    mpz_set_ui(e, z);
    gmp_printf("d = %Zd\n", e);
    // d = d//e
    mpz_fdiv_q(d,d,e);
    gmp_printf("c = z/d = %Zd\n", d);
    //encrypt d ie d = E(d) = E(d//z)
    pcs_encrypt(pk, hr_2, d, d);
    
    // b = y
    mpz_set_ui(b, y);
    // b = y//z
    mpz_fdiv_q(b,b,e);
    // b = -b = - y//z
    mpz_neg(b,b);
    // encrypt b ir b = E(b) = E(-y//z)
    pcs_encrypt(pk, hr_2, b, b);
    // d = d*b = E(d//z)*E(-y//z)
    mpz_mul (d, d, b);

    gmp_printf("a/d enc = %Zd\n", d);     // output: c = 126
    pcs_decrypt(vk, d, d);
    gmp_printf("a/d = %Zd\n", d);     // output: c = 126
    // Cleanup all data
    mpz_clears(a, b, c, NULL);
    pcs_free_public_key(pk);
    pcs_free_private_key(vk);
    hcs_free_random(hr);

    return 0;
}
std::vector<int> d_test_(int modsize=12) {
    pcs_public_key *pk = pcs_init_public_key();
    pcs_private_key *vk = pcs_init_private_key();
    hcs_random *hr = hcs_init_random();

    // Generate a key pair with modulus of size 2048 bits
    pcs_generate_key_pair(pk, vk, hr, modsize);
    //gmp_printf("%Zd\n", mpz_get_ui (hr->rstate->_mp_seed));
    //int(mpz_get_ui (hr->rstate->_mp_seed))
  return {1};
}
int test_(int x, int y, std::vector<unsigned long int> public_key, std::vector<unsigned long int> private_key)
{
    // initialize data structures
    //pcs_public_key *pk = pcs_init_public_key();
    //pcs_private_key *vk = pcs_init_private_key();
    hcs_random *hr = hcs_init_random();

    // Generate a key pair with modulus of size 2048 bits
    //pcs_generate_key_pair(pk, vk, hr, mod);
    pcs_public_key *pk = public_key_set(public_key);
    pcs_private_key *vk = private_key_set( private_key);
    // libhcs works directly with gmp mpz_t types, so initialize some
    mpz_t a, b, c;
    mpz_inits(a, b, c, NULL);

    mpz_set_ui(a, x);
    mpz_set_ui(b, y);
    gmp_printf("a = %Zd\nb = %Zd\n", a, b);
    pcs_encrypt(pk, hr, a, a);  // Encrypt a (= 50) and store back into a
    pcs_encrypt(pk, hr, b, b);  // Encrypt b (= 76) and store back into b
    gmp_printf("a = %Zd\nb = %Zd\n", a, b); // can use all gmp functions still

    pcs_ee_add(pk, c, a, b);  
    gmp_printf("c = %Zd\n", c);  // Add encrypted a and b values together into c
    pcs_decrypt(vk, c, c);      // Decrypt c back into c using private key
    gmp_printf("%Zd\n", c);     // output: c = 126

    // Cleanup all data
    mpz_clears(a, b, c, NULL);
    pcs_free_public_key(pk);
    pcs_free_private_key(vk);
    hcs_free_random(hr);

    return 0;
}

unsigned long int encrypt_s( long int x,  std::vector<unsigned long int> public_key)
{
    hcs_random *hr = hcs_init_random();
    // Generate a key pair with modulus of size 2048 bits
    //pcs_generate_key_pair(pk, vk, hr, mod);
    pcs_public_key *pk = public_key_set(public_key);
    
    mpz_t a;
    mpz_inits(a, NULL);

    mpz_set_ui(a, x);
    pcs_encrypt(pk, hr, a, a);  // Encrypt a (= 50) and store back into a
 
    pcs_free_public_key(pk);
    hcs_free_random(hr);

    return static_cast< unsigned long int >(mpz_get_ui(a));
}

unsigned long int decrypt_s(long int x,  std::vector<unsigned long int> private_key)
{
    // Generate a key pair with modulus of size 2048 bits
    //pcs_generate_key_pair(pk, vk, hr, mod);
    pcs_private_key *vk = private_key_set(private_key);
    
    mpz_t a;
    mpz_inits(a, NULL);
    mpz_set_ui(a, x);
    pcs_decrypt(vk, a,a);
    pcs_free_private_key(vk);

    return static_cast< unsigned long int >(mpz_get_ui(a));
}
torch::Tensor dummy_matmul(torch::Tensor E_x , 
                          torch::Tensor E_y, 
                          std::vector<unsigned long int> public_key, 
                          std::vector<unsigned long int> private_key,
                          int q = 1) {
  /*
    Takes E(x), E(y)  and computs E(x@y)
    */
  
  auto type = E_x.dtype();
  int x_n = E_x.size(0);
  int x_m = E_x.size(1);
  int y_n = E_y.size(0);
  int y_m = E_y.size(1);
  
  mpz_t a;
  mpz_inits(a, NULL);
  pcs_private_key *vk = private_key_set(private_key);
  hcs_random *hr = hcs_init_random();
  pcs_public_key *pk = public_key_set(public_key);

  auto s = torch::zeros({x_n, y_m}, type);
  #pragma omp parallel for private(i,j,k) shared(E_x,E_y,s)
    for(int i = 0; i < x_n; ++i) {
      auto E_x_i = E_x.index({i});
      for(int j = 0; j < y_m; ++j) {
        mpz_set_ui(a, mul_dec( E_x.index({i}), 
                            E_y.t().index({j}),
                            public_key, private_key, q ).sum().item<long>());
        pcs_encrypt(pk, hr, a, a);
        s[i][j] =long(mpz_get_ui(a)) ;
	    }
	}

  return s;
}
torch::Tensor sum_enc(torch::Tensor E_z, int axis, 
                    std::vector<unsigned long int> public_key,
                    std::vector<unsigned long int> private_key) {
    pcs_public_key *pk = public_key_set(public_key);
    pcs_private_key *vk = private_key_set( private_key);
    //pcs_private_key *vk = pcs_init_private_key();

    // Generate a key pair with modulus of size 2048 bits
    //pcs_generate_key_pair(pk, vk, hr, 2048);
    // libhcs works directly with gmp mpz_t types, so initialize some
    mpz_t a;
    mpz_inits(a, NULL);

    auto s = encrypt(decrypt(E_z,private_key).sum(axis), public_key);
    mpz_clears(a, NULL);
    pcs_free_public_key(pk);
  return s;
}
torch::Tensor retrieve_matmul(torch::Tensor E_x, 
                          torch::Tensor E_y, 
                          torch::Tensor E_xy,
                          std::vector<unsigned long int> public_key,
                          unsigned long int r_a,
                          unsigned long int r_b ) {
  /*
    Takes E(A.sum(0)), E(B.sum(1)), E(M) and  r_a, r_b  and computs E(M)*E(A.sum(0))^(-r_b)*E(B.sum(1))^(-r_b)*E(-r_a*r_b)
    */
  auto type = E_x.dtype();
  int x_n = E_xy.size(0);
  int x_m = E_xy.size(1);
  int y_n = E_xy.size(0);
  int y_m = E_xy.size(1);
  
  mpz_t a;
  mpz_inits(a, NULL);
  pcs_public_key *pk = public_key_set(public_key);

  auto s = torch::zeros({x_n, y_m}, type);
  #pragma omp parallel for 
  for(int i = 0; i < x_n; ++i) {
    torch::Tensor E_x_i = E_x[i].repeat(y_m); // sum over columns 
    s.index_put_({i}, retrieve_mul (E_x_i, E_y, E_xy.index({i}),
                                    public_key,
                                    r_a, r_b) ) ;
    }

  return s;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pcs_keys", &pcs_keys, "D_TEST");
  m.def("encrypt", &encrypt, "ENCRYPT");
  m.def("decrypt", &decrypt, "DECRYPT");
  m.def("test", &test, "TEST");
  m.def("test_", &test_, "TEST_");
  m.def("d_test_", &d_test_, "D_TEST_");
  m.def("modular_mul", &modular_mul, "MODULAR");
  m.def("mul_enc", &mul_enc, "MULENC");
  m.def("mul_enc2", &mul_enc2, "MULENC2");
  m.def("retrieve_mul", &retrieve_mul, "RETMUL");
  m.def("div_enc", &div_enc, "DIVENC");
  m.def("encrypt_s", &encrypt_s, "ENC_S");
  m.def("decrypt_s", &decrypt_s, "DEC_S");
  m.def("dummy_matmul", &dummy_matmul, "MATMUL");
  m.def("sum_enc", &sum_enc, "SUM_ENC");
  m.def("retrieve_matmul", &retrieve_matmul, "retrieve_matmul");
  m.def("modular_sum", &modular_sum, "modular_sum");
  m.def("modular_add", &modular_add, "modular_add");
  m.def("modular_sub", &modular_sub, "modular_sub");
  m.def("modular_pow", &modular_pow, "modular_pow");
  m.def("modular_pow", &modular_pow, "modular_pow");
  m.def("gmp_sign", &gmp_sign, "Gmp_sign");
}
