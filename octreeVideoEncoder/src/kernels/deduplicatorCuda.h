#include <layeredOctree/layeredOctreeContainerCuda.h>
#include "../encoding/encodingSequence.h"


void buildSimilarityLookupTableCuda(float* nearnessTable, std::vector<layer_ptr_type>& population, int layer, LayeredOctreeContainerCuda<OctreeProcessingPayload<octreeColorType>>* container);