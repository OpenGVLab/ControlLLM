from typing import List, Union
import ast

from cllm.agents.base import Action


class Parser:
    def parse(self, plan) -> List[Action]:
        # ignore indent
        input = '\n'.join([line.strip() for line in plan.split('\n')])
        actions = []
        for stmt in ast.parse(input).body:
            if isinstance(stmt, ast.Assign):
                assign: ast.Assign = stmt
                output: ast.Name = assign.targets[0]
                func_call: ast.Call = assign.value
                func_name: ast.Name = func_call.func
                kwargs: List[ast.keyword] = func_call.keywords
                args = {}
                for kwarg in kwargs:
                    k = kwarg.arg
                    if isinstance(kwarg.value, ast.Name):
                        v = kwarg.value.id
                    else:
                        v = ast.literal_eval(kwarg.value)
                    args[k] = v
                action = Action(tool_name=func_name.id, outputs=[output.id], inputs=args)
                actions.append(action)
        return actions


class Compiler:
    def __init__(self):
        self.parser = Parser()

    def compile(self, plan: Union[str, List[Union[Action, str]]]) -> List[Action]:
        """ The input could be a plain string, a list of structured `Action` 
            or combination of structured `Action` or unstructured action string.
        """
        actions = self.parse(plan)
        actions = self.correct(actions)
        return actions

    def parse(self, plan) -> List[Action]:
        if isinstance(plan, str):
            return self.parser.parse(plan)

        actions = []
        for action in plan:
            if isinstance(action, str):
                action = self.parser.parse(action)[0]
            actions.append(action)

        return actions

    def correct(self, actions):
        return actions
