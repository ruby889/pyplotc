
from pycparser import c_parser, c_ast
text = r"""
    void write_to_file()
    {
        for (int i = 0; i < NUM_ARM; i++)
        {
            for (int j = 0; j <= JOINTS_PER_ARM; i++)
            {
                fprintf(fwrite_data, "%f ", ABC->set_velocity[i][j]);
            }
        }
        
        
        for (int i = 0; i < NUM_JOINTS; i++)
        {
            fprintf(fwrite_data, "%f ", read_position[i]);
        }

        fprintf(fwrite_data, "%d ", status);
        fprintf(fwrite_data, "%d %d ", receiver_communication_time, receiver_running_time);
        fprintf(fwrite_data, "\n");
    }
"""
# text = r"""
#     typedef int Node, Hash;

#     void HashPrint(Hash* hash, void (*PrintFunc)(char*, char*))
#     {
#         unsigned int i;

#         if (hash == NULL || hash->heads == NULL)
#             return;

#         for (i = 0; i < hash->table_size; ++i)
#         {
#             Node* temp = hash->heads[i];

#             while (temp != NULL)
#             {
#                 PrintFunc(temp->entry->key, temp->entry->value);
#                 temp = temp->next;
#             }
#         }
#     }
# """


# ast.show()
# print("////////////////////////////")
# ast.ext[2].show()

# print("////////////////////////////AAAAAAA")

# function_body = ast.ext[2].body
# for_stmt = function_body.block_items[2]
# for_stmt.show()

parser = c_parser.CParser()
ast = parser.parse(text, filename='<none>')
for block in ast.ext[0].body.block_items:
    print("///////////////////////A")
    block.show()
    if isinstance(block, c_ast.For):
        print("For loop found:")
        block.show()
        print("Initialization:")
        block.init.show()
        print("Condition:")
        block.cond.show()
        print("Next:")
        block.next.show()
        print("Body:")
        if isinstance(block.stmt, c_ast.Compound):
            for stmt in block.stmt.block_items:
                stmt.show()
        else:
            block.stmt.show()
    elif isinstance(block, c_ast.FuncCall):
        print("Function call found:")
        block.show()
        print("Function name:", block.name.name)
        if block.args:
            print("Arguments:")
            for arg in block.args.exprs:
                arg.show()