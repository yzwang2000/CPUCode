# CPUCode
此仓库是本人在秋招准备过程中关于 `CPU SIMD` 编程技巧的总结

## x64 架构下的寄存器模型
在一个现有的平台学习使用汇编的时候，要首先熟悉其寄存器组
![x64 寄存器组](./figure/寄存器模型.png)

`x86` 的汇编语言格式通常有两种, 一种是 `Intel`, 一种是 `AT&T`。单个文件生成汇编代码, `gcc -fomit-frame-pointer -fverbose-asm -S main.cpp -o main.asm`。
![汇编语言格式](./figure/汇编语言格式.png)

## 常见的优化手法
* CPU 端的优化手段, 就是尽可能地让指令都是矢量化指令(也就是让运算都是矢量化的运算)。单个指令处理多个数据的技术称为 `SIMD`, 可以大大增加计算密集型程序的吞吐量。通常认为利用同时处理 4 个 `float` 的 `SIMD` 指令可以加速 4 
```C++
float func(float a, float b){
    return a+b;
}

// 对应的汇编语言为
func(float, float):
    addss   xmm0, xmm1
    ret
```
* 其中 `xmm` 寄存器有 128 位宽, 可以容纳 4 个 `float` 或者 2 个 `double`。上面的代码中只用到了 `xmm` 的低 32 位用于存储 1 个 `float`。其中的 `addss` 可以拆分成 `add` `s` `s`。`add` 表示执行加法操作。 第一个 `s` 表示标量(scalar), 即只对应 `xmm` 的最低位进行运算, 也可以是 `p` 表示矢量(packed), 一次对 `xmm` 的所有位进行运算。第二个 `s` 表示单精度浮点数(single), 即 `float`, 也可以是 `d` 表示双精度浮点数(double)。所以 `addss` 一个 `float` 加法。`addsd` 一个 `double` 加法。`addps` 四个 `float` 加法。`addpd` 两个 `double` 加法。

![scalar 和 packed](./figure/scalar_packed.png)

### 多利用 `constexpr` 强迫编译器在编译期求值
* 编译器能够自动完成许多优化, 代数化简, 常量折叠
```C++
// 编译器进行代数化简, 编译时优化
int func(int a, int b){
    int c = a+b;
    int d = a-b;
    return (c+d)/2;
}

func(int, int):
    mov     eax, edi
    ret

// 编译器进行常量折叠, 编译时优化
int func(int a, int b){
    int ret = 0;
    for(int i=1; i<=100; i++){
        ret += i;
    }
    return ret;
}

func(int, int):
    mov     eax, 5050
    ret
```
* 但是有问题是对于存储在`堆`上的, 是不利于优化的。存储在`栈`上的, 是利于优化的。存储在 `堆` 上的有: `vector`, `map`, `set`, `string`, `unique_ptr`, `shared_ptr`, `weak_ptr`。存储在`栈`上的有: `array`, `tuple`, `pair`, `optional`, `string_view`。
* 对于 `栈` 上的复杂的代码, 编译器会放弃优化, 但是我们可以使用 `constexpr` 强迫其进行编译时优化。但是 `constexpr` 函数中使用非 `constexpr` 的容器(`vector`, `map`, `set`, `string`, ...), 函数会转变为普通的函数。
```C++
// 利用 constexpr 强迫其进行编译时优化
#include<array>

template <int N>
constexpr int func_impl() {
    std::array<int, N> arr{};
    for (int i = 1; i <= N; i++) {
        arr[i - 1] = i;
    }
    int ret = 0;
    for (int i = 1; i <= N; i++) {
        ret += arr[i - 1];
    }
    return ret;
}

int func() {
    constexpr int ret = func_impl<50000>();
    return ret;
}

func():
    push    rbp
    mov     rbp, rsp
    mov     DWORD PTR [rbp-4], 1250025000
    mov     eax, 1250025000
    pop     rbp
    ret
```
* `const` 并未区分出编译时常量和运行时常量, `constexpr` 限定在编译期常量。`constexpr` 可以修饰变量和函数。`constexpr` 修饰的函数, 如果其传入的参数可以在编译时期计算出来, 这个函数就会产生编译时期的值。如果不能的话, `constexpr` 修饰的函数就和普通函数一样。现如今, `const` 表示运行时不能被修改的常量。`constexpr` 修饰变量的时候, 要求其初始化的值必须是在编译期可确定的。
```C++
const int a = 5;  // 运行时确定的常量
constexpr int b = 10;  // 编译期确定的常量表达式
```

### 利用内联优化
* 定义在同一个文件的函数才可以被内联, 定义在不同文件中, 编译器看不见函数体里的内容, 没有办法进行内联。
```C++
// other 定义在其他的文件中, 这里只是声明, 其函数体是空的。但是我们可以从汇编代码看到, 编译器并没有实现内联。
int other(int a);

int func(){
    for(int i=0; i<100; ++i){
        other(22);
    }
}

func():
    sub     rsp, 8
.L2:
    mov     edi, 22
    call    other(int)
    jmp     .L2
```
* 所以我们为了效率尽量把 `常用函数的定义` 放在头文件中, 声明为 `static` 或者 `inline`, 这样调用他们的时候编译器磁能看到他们的函数体, 从而有机会内联。内不内联只取决于`同文件且函数体足够小`, 与是否加 `inline` 没有关系。

### 利用 __restrict 关键字
* `__restrict` 是 C99 的标准关键字, 但并不是 C++ 的关键字, 但是绝大多数主流编译器都支持。它用于向编译器指示指针所指向的对象具有特定的访问限制，即被标记为 `__restrict` 的指针是 `该对象的唯一访问途径`。
```C++
// 为什么会生成这么复杂的汇编代码, 是因为编译器不确定这些指针是否有重叠。
// 在 a, b, c 指向均无重叠的时候, c 指向的值与 b 的值相同 (第一句就可以省略的)。
// 在 b=c, 但与 a 无重叠的时候, c 指向的值为 a (第一句就不可以优化掉)。
// 所以我们可以使用 __restrict 来限制非 const 修饰的指针, 来告诉编译器具有写入权限的指针是独享其资源的。
void func(const int *a, const int *b, int*c){
    *c = *a;
    *c = *b;
}

func(int const*, int const*, int*):
        push    rbp
        mov     rbp, rsp
        mov     QWORD PTR [rbp-8], rdi
        mov     QWORD PTR [rbp-16], rsi
        mov     QWORD PTR [rbp-24], rdx
        mov     rax, QWORD PTR [rbp-8]
        mov     edx, DWORD PTR [rax]
        mov     rax, QWORD PTR [rbp-24]
        mov     DWORD PTR [rax], edx
        mov     rax, QWORD PTR [rbp-16]
        mov     edx, DWORD PTR [rax]
        mov     rax, QWORD PTR [rbp-24]
        mov     DWORD PTR [rax], edx
        nop
        pop     rbp
        ret

void func(const int * a, const int * b, int* __restrict c){
    *c = *a;
    *c = *b;
}

func(int const*, int const*, int*):
    mov     eax, DWORD PTR [rsi]
    mov     DWORD PTR [rdx], eax
    ret
```
* 所有 `非 const` 的指针都声明 `__restrict`。
* 对于 `std::vector` 也有指针别名的问题, 但是不能使用 `__restrict` 来解决。但是我们可以通过 `#pragma GCC ivdep` 预处理指令来处理。
```C++
#include <vector>

void func(std::vector<int>&a, std::vector<int>&b, std::vector<int>&c){
    #pragma GCC ivdep
    for(int i=0; i<1024; ++i){
        a[i] = b[i] + 1;
    }
}
```

### 矢量化
* CPU 也能利用 `SIMD` 的矢量化指令来进行优化。但是通常要注意几个问题, 编译器才能自动的利用 `SIMD` 指令来进行优化。
    * 访问数组等数据结构时, 尽可能的连续访问。
    ```C++
    // 对数据的跳跃访问, 我们可以注意到汇编代码中, 只有部分是矢量化成功了(因为有 ps 和 ss)
    void func(float *a){
        for(int i=0; i<1024; ++i){
            a[i*2]+=20;
        }
    }

    func(float*):
            movss   xmm2, DWORD PTR .LC1[rip]
            mov     rax, rdi
            lea     rdx, [rdi+8160]
            shufps  xmm2, xmm2, 0
    .L2:
            movups  xmm0, XMMWORD PTR [rax]
            movups  xmm3, XMMWORD PTR [rax+16]
            add     rax, 32
            shufps  xmm0, xmm3, 136
            addps   xmm0, xmm2
            movaps  xmm1, xmm0
            movss   DWORD PTR [rax-32], xmm0
            shufps  xmm1, xmm0, 85
            movss   DWORD PTR [rax-24], xmm1
            movaps  xmm1, xmm0
            unpckhps        xmm1, xmm0
            shufps  xmm0, xmm0, 255
            movss   DWORD PTR [rax-8], xmm0
            movss   DWORD PTR [rax-16], xmm1
            cmp     rax, rdx
            jne     .L2
            movss   xmm0, DWORD PTR .LC1[rip]
            movss   xmm1, DWORD PTR [rdi+8160]
            addss   xmm1, xmm0
            movss   DWORD PTR [rdi+8160], xmm1
            movss   xmm1, DWORD PTR [rdi+8168]
            addss   xmm1, xmm0
            movss   DWORD PTR [rdi+8168], xmm1
            movss   xmm1, DWORD PTR [rdi+8176]
            addss   xmm1, xmm0
            addss   xmm0, DWORD PTR [rdi+8184]
            movss   DWORD PTR [rdi+8176], xmm1
            movss   DWORD PTR [rdi+8184], xmm0
            ret
    .LC1:
            .long   1101004800
    
    // 修改为连续读取以后, 矢量化大成功(基本都是 ps)
    void func(float *a){
        for(int i=0; i<1024; ++i){
            a[i]+=20;
        }
    }

    func(float*):
            movss   xmm1, DWORD PTR .LC1[rip]
            lea     rax, [rdi+4096]
            shufps  xmm1, xmm1, 0
    .L2:
            movups  xmm0, XMMWORD PTR [rdi]
            add     rdi, 16
            addps   xmm0, xmm1
            movups  XMMWORD PTR [rdi-16], xmm0
            cmp     rax, rdi
            jne     .L2
            ret
    .LC1:
            .long   1101004800
    ```
    * 我们要给编译器提示, 如以下代码中提示编译器, `N` 是能够被 4 整除的, 放心优化, 没有边界问题。
    ```C++
    void func(float *a, std::size_t N){
        N = n/4 * 4;
        for(int i=0; i<N; ++i){
            a[i*2]+=20;
        }
    }

    ```

    * 对于结构体, 我们要善于使用对齐或者填充操作 `alignas` 对齐关键字。
    ```C++
    // alignas(express) 设置的的对齐值 `express` 必须是 2 的幂次方。在结构体中, 会以占用最大字节的成员变量的字节数和 express 的最大值做为内存对齐的基数
    // 一个实例化对象占用 16 个字节, 因为是以 sizeof(double) 为基数的
    struct alignas(4) struct_Test2
    {
        char c;
        int  i;
        double d;
    };
    
    // 一个实例化对象占用 32 个字节, 以 32 为基数。
    struct alignas(32) struct_Test3
    {
        char c;
        int  i;
        double d;
    };
    ```
    * 尽量将结构体的大小设置为 2 的整数幂(2, 4, 8, 16, 32, 64)。如果结构体的大小不是 2 的整数幂, 往往会导致 `SIMD` 优化失败。
    ```C++
    // 这里加了 alignas, 在汇编中指令减少了很多
    struct alignas(16) MyVec {
    float x;
    float y;
    float z;
    };

    MyVec a[1024];

    void func() {
        for (int i = 0; i < 1024; i++) {
            a[i].x *= a[i].y;
        }
    }
    ```
    * 结构的两种布局 `AOS(Array of Struct)` 和 `SOA(Struct of Array)`。`AOS` 是单个对象的属性紧挨着存, `SOA` 属性分离存储在多个数组中。`AOS` 需要结构体对齐到 2 的整数次幂才高效, `SOA` 就不需要了, 因为 `SOA` 天生就能对齐到 2 的整数次幂。但是 `SOA` 的问题就是要时刻确保各个属性数组元素个数相等。
    *  当程序中只需要访问某个结构体的一个成员变量的时候, 这时候 `SOA` 的优势就会特别明显。
    ```C++
    // Array of structures AOS
    struct Particle {float x, y, z, w};
    Particle particles[1000];

    // Structure of arrays SOA
    struct Particles {
        float x[1000];
        float y[1000];
        float z[1000];
        float w[1000];
    };
    ```

### 循环的优化
* 对于程序中循环的优化, 通常有以下几点
    * 循环中的 `if` 语句尽可能挪到外边来
    ```C++
    void func(float *__restrict a, float *__restrict b, bool is_mul) {
        for (int i = 0; i < 1024; i++) {
            if (is_mul) {
                a[i] = a[i] * b[i];
            } else {
                a[i] = a[i] + b[i];
            }
        }
    }

    void func(float *__restrict a, float *__restrict b, bool is_mul) {
        if(is_mul){
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] * b[i];
        }
        }else{
            a[i] = a[i] * b[i];
        }
    }
    ```
    * 循环中的不变量挪到循环外边来(减少计算), 其实编译器也能自己识别, 我们将其打上括号。
    ```C++
    void func(float *__restrict a, float *__restrict b, float dt) {
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] + b[i] * dt * dt;
        }
    }

    void func(float *__restrict a, float *__restrict b, float dt) {
        float dt2 = dt * dt;
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] + b[i] * dt2;
        }
    }

    void func(float *__restrict a, float *__restrict b, float dt) {
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] + b[i] * (dt * dt);
        }
    }
    ```
    * 对小的循环可以使用 `#pragma GCC unroll n` 来进行循环展开。对于大的循环最好不要进行 `unroll`, 会造成指令缓存和寄存器压力, 反而变慢。
    ```C++
    // 如果不能够被整除的时候, unroll 能够自动处理边界条件
    void func(float *a){
        #pragma GCC unroll 4
        for(int i=0; i<1024; ++i){
            a[i]=1;
        }
    }
    ```

### 数学运算的优化
* 对于浮点数来说, 除法尽量变成乘法;
```C++
float func(float a)
{
    return a/2;
}

float func(float a)
{
    return a*0.5f;
}
```
* 对于整数来说, 如果乘以或者除以 2 的整除次幂, 尽量抓换成移位的操作(保证被操作数是无符号数或者有符号正数)。
```C++
int func(std::size_t N){
    N >>= 1;  // N = N/2;
}
```
* 对于整数的取余操作, 如果是对 2 的整数次幂取余, 尽量转换为 & 操作(保证被操作数是无符号数或者有符号正数)。
```C++
void func(std::size_t N)
{
    N &= (4-1);  // N = N % 4;
}
```
* 如果能够保证程序中不会出现 `NaN` 和 `Inf`, 可以增加编译器参数 `-ffast-math`, 让 GCC 更加大胆地尝试浮点运算的优化, 有时能带来 2 倍左右的速度提升。