
class Node():
    def __init__(self, index):
        self.index = index
        #self.data = data
        self.left = None
        self.right = None
        

class SimpleCacheQueue():
    """
    Implementation of a simple cache which maintains a queue for eviction, and
    a dictionary (Hash Map) for fast key retrieval. Most recently used item is at
    the left (head) of the queue, least recently used is at the right (tail).
    """
    def __init__(self, cache_size):
        self.MAX_SIZE = cache_size
        self.lookup = dict()
        self.dummy = Node(-1)
        self.head = self.dummy.left
        self.tail =  self.dummy.left
        
    def evict(self, index_to_remove=None):
        if index_to_remove is None:
            # Remove tail node
            old_tail = self.tail
            old_index = self.tail.index
            self.tail = self.tail.left
            if self.tail:
                self.tail.right = None
            try:
                self.lookup.pop(int(old_index))
            except:
                pass
            del old_tail
        else:
            victim = self.lookup[int(index_to_remove)]
            if victim.left is not None:
                victim.left.right = victim.right
            if victim.right is not None:
                victim.right.left = victim.left
            self.lookup.pop(int(index_to_remove))
            del victim
            return index_to_remove
            
        return old_index
        
    def insert(self, index):
        new_node = Node(index)
        if self.head is None:
            self.head = self.tail = new_node

        else:
            # A newly inserted item goes to head of cache queue
            self.head.left = new_node
            new_node.right = self.head
            self.head = self.head.left
            
        
        self.lookup[index] = new_node
        assert self.tail is not None

        
