#define NO_NODE -1

template <typename T>
class ChunkedOctree {
    int offsets[8] {
        NO_NODE, NO_NODE, NO_NODE, NO_NODE,
        NO_NODE, NO_NODE, NO_NODE, NO_NODE
    };
    T payload;
    // Makes programming a bit easier
    uint8_t layer = 0;
    // Number of children
    uint8_t childCount = 0;
    // Bit flags for the children
    uint8_t childFlags = 0;
public:
    ChunkedOctree(T payload, uint8_t layer);

    ChunkedOctree<T>* getChildByCoords(int x, int y, int z) const {
        return this->getChildByIdx(ADDRESS_OCTREE(x, y, z));
    }
    ChunkedOctree<T>* getChildByIdx(int idx) const {
        return this->children[idx];
    }

    int getHashKey();

    T* getPayload() {
        return &this->payload;
    }

    uint8_t getLayer() {
        return this->layer;
    }

    void setChild(int childIdx, int idx) {
        if(this->offsets[idx] != NO_NODE) {
            throw "Child already existed";
        } else {
            this->childCount++;
        }
        this->offsets[idx] = childIdx;
    }
    uint8_t getChildCount() const {
        return this->childCount;
    }
};

template <typename T>
ChunkedOctree<T>::ChunkedOctree(T payload, uint8_t layer) : payload(payload), layer(layer) {

}
