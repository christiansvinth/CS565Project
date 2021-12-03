
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
            if(self.tail is None):
                print("No tail!")
            old_tail = self.tail
            old_index = self.tail.index
            self.tail = self.tail.left
            if self.tail:
                self.tail.right = None
                
            self.lookup.pop(old_tail.index)
            del old_tail
        else:
            victim = self.lookup(index)
            victim.left.right = victim.right
            victim.right.left = victim.left
            self.lookup.pop(index)
            del victim
            
        return old_index
        
    def insert(self, index):
        new_node = Node(index)
        if self.head is None:
            self.head = self.tail = new_node
            #self.head.right = self.tail
            #self.tail.left = self.head
        else:
            # A newly inserted item goes to head of cache queue
            self.head.left = new_node
            new_node.right = self.head
            self.head = self.head.left
            
        
        self.lookup[new_node.index] = new_node
        assert self.tail is not None

        
