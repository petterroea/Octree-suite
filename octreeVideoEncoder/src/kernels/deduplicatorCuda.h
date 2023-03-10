#include "../layeredOctree/layeredOctreeContainerCuda.h"
#include "../encoding/encodingSequence.h"


void buildSimilarityLookupTableCuda(float* nearnessTable, int populationSize, int layer, LayeredOctreeContainerCuda<octreeProcessingPayload>& container);