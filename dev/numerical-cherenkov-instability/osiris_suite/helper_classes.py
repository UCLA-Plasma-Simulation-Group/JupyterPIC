import sys 




# data container for the params. example usage: 
# x = AttrDict()
# x.a = 2
# x['a'] --> returns 2
# x.a --> returns 2 
# see https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute



class OsirisSuiteError( Exception ) : 
    ... 




class AttrDict( dict ):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def clear( self ) :
        for key in self.keys() :
            del self[ key ] 



class RecursiveAttrDict( AttrDict ) : 
    '''
        this is a subclass of the attrdict which gives support for recursive attrdict operations  
    '''
                
    # def __str__( self, depth, accumulated_str, print_vals = False ) : 
    def __str__( self, print_vals = False ) : 
        return attrdict2str( self, '', [''], print_vals )[0]


    def prune( self, print_status = False ) : 
        return recursively_prune_attrdict( self, None, None, print_status )


    def find_subtree( self, subtree_name ) : 
        return recursively_find_subtree( subtree_name, self )

     


def attrdict2str( attrdict, dash_str, accumulated_str, print_vals ) : 


    #accumulated_str = ''

    # print( attrdict.keys() )

    try : 
        keys = sorted( attrdict.keys() )
        # print( keys ) 
    except : 
        return ''


    for key in keys : 

        # print ('adding key: ' + key )
        
        val = attrdict[ key ]

        # print( type( val ) ) 

        currline = dash_str + ' ' +  key + '\n'

        accumulated_str[0] += currline 

        if isinstance( val, AttrDict ) : 

            # print( 'adding currline: '  + currline )
            
            # accumulated_str_prev  = accumulated_str

            # print( 'added currline: '  + currline )

            # print( 'accumulated_str: ' + accumulated_str[0] )

            # accumulated_str += attrdict2str( val, depth + 1, accumulated_str, print_vals )

            new_dash_str =  ( ' ' * ( len(dash_str) + 1 ) 
                                + '-' * len( key ) 
                                + '->' )

            attrdict2str( val, new_dash_str, accumulated_str, print_vals )

            # print( 'new accumulated_str: ' + accumulated_str[0] )

        else : 
            if print_vals :
                accumulated_str += str( val )


    return accumulated_str




     

def recursively_prune_attrdict( current_attrdict, current_key, 
                                parent_attrdict, print_status = False ) : 
    '''
        iteratively remove all empty attr dicts
    '''

    # use a list here because .items() and .keys() return generators 
    # this avoids RuntimeError: dictionary changed size during iteration

    for key in list( current_attrdict.keys() ) : 
        
        val = current_attrdict[ key ]

        if isinstance( val, AttrDict ) : 
            recursively_prune_attrdict( val, key, current_attrdict, print_status )
    
    # perform the prune
    if parent_attrdict is not None and len( current_attrdict ) == 0 :

        if print_status : 
            print( 'Pruning key: %s' % current_key )

        del parent_attrdict[ current_key ] 



def recursively_find_subtree( subtree_name, current_dict ) :
    
    if current_dict is None : 
        current_dict = self.data

    for name, subdict in current_dict.items() : 

        if not isinstance( subdict, AttrDict ) : 
            continue 

        # return branch if we found it
        if name == subtree_name : 
            return subdict 

        # otherwise search all subtrees  
        else : 
            ret = recursively_find_subtree( subtree_name, subdict )

            if ret is not None : 
                return ret

    # only reached if no matching subtree is found. 
    return None 

