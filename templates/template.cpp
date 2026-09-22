// ─────────────────────────────────────────────────────────────────────────────
// Competitive programming template — works on Codeforces, AtCoder, CSES, local
// Compile locally:  g++ -std=c++20 -O2 -DLOCAL template.cpp -o sol
// Submit as-is (no -DLOCAL → dbg compiles to nothing, everything else neutral)
// ─────────────────────────────────────────────────────────────────────────────
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

// ── typedefs ─────────────────────────────────────────────────────────────────
using ll  = long long;
using ull = unsigned long long;
using ld  = long double;
using vi  = vector<int>;
using vll = vector<ll>;
using pii = pair<int, int>;
using pll = pair<ll, ll>;
template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<class T> using max_heap = priority_queue<T>;

// order-statistics tree (find_by_order / order_of_key)
template<class K>
using oset = tree<K, null_type, less<K>, rb_tree_tag,
                  tree_order_statistics_node_update>;

// ── constants ────────────────────────────────────────────────────────────────
constexpr ll  MOD = 1'000'000'007;      // 998'244'353 for NTT problems
constexpr ll  INF = 4'000'000'000'000'000'000;  // > max path/answer, no overflow
constexpr ld  EPS = 1e-9;

// ── debug (LOCAL builds only — submissions get nothing) ──────────────────────
#ifdef LOCAL
template<class T> void __pr(const T &v) { cerr << v; }
template<class T, class U> void __pr(const pair<T, U> &p) { cerr << '(' << p.first << ',' << p.second << ')'; }
template<class T> void __pr(const vector<T> &v) { cerr << '['; for (const auto &x : v) { __pr(x); cerr << ' '; } cerr << ']'; }
void __dbg(const char *n) { cerr << "  " << n << '\n'; }
template<class F, class... R>
void __dbg(const char *n, F &&f, R &&...rest) {
    const char *c = strchr(n, ',');
    cerr << "  " << string(n, c ? c : n + strlen(n)) << " = ";
    __pr(forward<F>(f));
    if (c) __dbg(c + 1, forward<R>(rest)...);
    else cerr << '\n';
}
#define dbg(...) do { cerr << "\033[32m[dbg]\033[0m"; __dbg(#__VA_ARGS__, __VA_ARGS__); } while (0)
#else
#define dbg(...) ((void)0)
#endif

// ── fast IO ──────────────────────────────────────────────────────────────────
void io() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
#ifdef LOCAL
    freopen("input.txt", "r", stdin);   // reads input.txt next to binary
#endif
}

// ── helpers ──────────────────────────────────────────────────────────────────
template<class T> T &ckmax(T &a, const T &b) { return a < b ? a = b : a; }
template<class T> T &ckmin(T &a, const T &b) { return a > b ? a = b : a; }

// ─────────────────────────────────────────────────────────────────────────────
void solve() {
    int n;
    cin >> n;
    vi a(n);
    for (int &x : a) cin >> x;

    oset<int> s(a.begin(), a.end());
    ll ans = *s.find_by_order(n / 2);      // median (example usage)
    dbg(n, a, ans);                        // prints n, a, ans in LOCAL builds

    cout << ans << '\n';                   // '\n' > endl (no flush per line)
}

int main() {
    io();
    int t = 1;
    // cin >> t;                           // uncomment for multiple test cases
    while (t--) solve();
    return 0;
}
