float aboveZero;
float belowZero;
float x;

int foo() {
#pragma scop

  float x = 1.0f;
  float mulfactor = x > 0 ? aboveZero : belowZero;

#pragma endscop
  return 0;
}