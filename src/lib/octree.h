#define ADDRESS_OCTREE(x, y, z) (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)

template <typename T>
class Octree {
    Octree<T>* children[8];
    T payload;
public:
    Octree(T payload);
    ~Octree();

    Octree<T>* getChild(int x, int y, int z);
    T* getPayload();
    bool setChild(Octree<T>* child, int x, int y, int z);
};