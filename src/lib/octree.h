#define ADDRESS_OCTREE(x, y, z) (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)

template <typename T>
class Octree {
    Octree<T>* children[8] {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    T payload;
    uint8_t childCount = 0;
public:
    Octree(T payload);
    ~Octree();

    Octree<T>* getChildByCoords(int x, int y, int z) {
        return this->getChildByIdx(ADDRESS_OCTREE(x, y, z));
    }
    Octree<T>* getChildByIdx(int idx) {
        return this->children[idx];
    }

    T* getPayload() {
        return &this->payload;
    }
    bool setChild(Octree<T>* child, int idx) {
        bool removed_existing = false;
        if(this->children[idx] != nullptr) {
            delete this->children[idx];
            removed_existing = true;
        } else {
            this->childCount++;
        }
        this->children[idx] = child;
        return removed_existing;
    }
    uint8_t getChildCount() {
        return this->childCount;
    }
};

template <typename T>
Octree<T>::Octree(T payload) {
    this->payload = payload;
}

template <typename T>
Octree<T>::~Octree() {
    for(int i = 0; i < 8; i++) {
        if(children[i] != nullptr) {
            delete children[i];
        }
    }
}
