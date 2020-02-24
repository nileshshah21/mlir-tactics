class NativeCodeCall<string expr> {
  string expression = expr;
}

class Tactics <string p, NativeCodeCall c> {
    string pattern = p;
    NativeCodeCall builder = c;
}

def transposeT : Tactics<"C(i,j,k) = A(i,l) * B(j,l,k)", NativeCodeCall<"transpose(B, {1, 0, 2})">>;
