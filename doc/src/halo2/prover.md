# Prover

证明阶段的输入为 `Vec<Instance>` 和 `Vec<ConcreteCircuit>` ，其中 `Instance` 包含公共输入，`ConcreteCircuit` 包含隐私输入，
证明写入 `T: TranscriptWrite`

```rust
pub fn create_proof<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
    ConcreteCircuit: Circuit<C::Scalar>,
>(
    params: &Params<C>,                           // 包括多项式提交所需的椭圆曲线点等
    pk: &ProvingKey<C>,                           // 包括fixed列的值、permutation信息
    circuits: &[ConcreteCircuit],
    instances: &[&[&[C::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>

```

以下说明均采用 Fiat-Shamir 变换前的视角，并且常常混用列、列多项式、列多项式的各种表示。

代码中频繁出现的 `EvaluationDomain` 类型保存了做多项式运算必须的数据，例如数论变换所需的单位根及其阶数、数论变换所需的除数、
消灭多项式（vanishing polynomial）\\(X^n - 1\\) 的拉格朗日基表示、等等。

Halo2 中的 "degree" 常常有不同的含义
- 一般表示表示多项式的阶
- 约束系统（constrain system）的阶表示将列多项式（fixed，advice 或者 instance 列对应的多项式）视为一次、约束多项式的次数。
因而，约束多项式的阶为约束系统的阶乘以电路的行数 \\(n\\)。

显然，\\(n\\) 个拉格朗日基不足以表示约束多项式，因而需要计算大多数出现的多项式的扩展拉格朗日基表示，
即用更多的拉格朗日基表示。

Halo2 中的“消灭多项式（vanishing polynomial）”指称不明。以下说明中将统一使用“消灭多项式”的名称，但是会指派不同的符号。

## 处理 `instances`

在 Halo2 的术语中，instance 即表示公开输入。

```rust
    let instance: Vec<InstanceSingle<C>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<C>, Error> {
            let instance_values = ...;
            let instance_commitments_projective: Vec<_> = instance_values
                .iter()
                // 此处使用MSM
                .map(|poly| params.commit_lagrange(poly, Blind::default()))  
                .collect();
            
            ...

            // 发送MSM结果
            for commitment in &instance_commitments {
                transcript.common_point(*commitment)?;
            }

            let instance_polys: Vec<_> = ...;
            let instance_cosets: Vec<_> = ...;

            Ok(InstanceSingle {
                instance_values,      // 拉格朗日基表示
                instance_polys,       // 系数表示
                instance_cosets,      // 扩展拉格朗日基表示
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
```

在这一步，`instances` 被表示成多项式 \\(A(X)\\) ，然后
- \\(A(X)\\) 被多项式提交、发送给客户端
- \\(A(X)\\) 经过数论变换转换为系数形式，保存备用
- \\(A(X)\\) 的系数形式再经过数论变化，得到扩展拉格朗日基下的表示，保存备用

## 处理 `advices`

在 Halo2 术语中，advice 表示隐私输入和电路内部状态。

```rust
    let advice: Vec<AdviceSingle<C>> = circuits
        .iter()
        .zip(instances.iter())
        .map(|(circuit, instances)| -> Result<AdviceSingle<C>, Error> {
            struct WitnessCollection<'a, F: Field> {
                k: u32,
                pub advice: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
                instances: &'a [&'a [F]],
                usable_rows: RangeTo<usize>,
                _marker: std::marker::PhantomData<F>,
            }

            impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
                ...
            }

            // 因为最后几行需要保留给盲化因子，所以并不是所有的行都可用
            let unusable_rows_start = params.n as usize - (meta.blinding_factors() + 1);

            let mut witness = /* 空 `WitnessCollection` */;

            // 执行电路的 `synthesize` 方法来获得电路的隐藏状态，当然也包括可能存在的隐私输入
            ConcreteCircuit::FloorPlanner::synthesize(
                &mut witness,
                circuit,
                config.clone(),
                meta.constants.clone(),
            )?;

            let mut advice = batch_invert_assigned(witness.advice);

            // Add blinding factors to advice columns
            ...

            // Compute commitments to advice column polynomials
            let advice_blinds: Vec<_> = /* 一些随机数 */;
            let advice_commitments_projective: Vec<_> = advice
                .iter()
                .zip(advice_blinds.iter())
                // 此处MSM
                .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
                .collect();
            ...

            // 发送MSM结果
            for commitment in &advice_commitments {
                transcript.write_point(*commitment)?;
            }

            let advice_polys: Vec<_> = ...;

            let advice_cosets: Vec<_> = ...;

            Ok(AdviceSingle {
                advice_values: advice,  // 拉格朗日基形式
                advice_polys,           // 系数形式
                advice_cosets,          // 扩展拉格朗日基形式
                advice_blinds,          // 提交多项式时使用的盲化因子
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
```

处理 advice 的过程类似于 instance ，不同处仅仅在于 advice 的值首先需要执行电路才能获得。

## 合并 PLOOKUPS 约束涉及的列

Halo2 协议基于 PLOOKUPS 协议提供了查表约束（ lookup ）。

给定关于输入列
    \\[a_1(X), a_2(X), ..., a_n(X)\\]
的表达式 \\(A(X)\\), 及关于列表列
    \\[s_1(X), a_2(X), ..., s_m(X)\\]
的表达式 \\(S(X)\\)，查表约束保证 \\(A(X)\\) 中任一行均能在 \\(S(X)\\) 中找到。

在一个电路中，可能有多个这样的约束。Halo2 通过挑战随机数将它们合并。

首先，验证着发送挑战随机数 \\(theta\\):
```rust
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();
```

然后证明者执行
```rust
    let lookups: Vec<Vec<lookup::prover::Permuted<C, _>>> = instance_values
        .iter()
        .zip(instance_cosets.iter())
        .zip(advice_values.iter())
        .zip(advice_cosets.iter())
        .map(|(((instance_values, instance_cosets), advice_values), advice_cosets)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        &value_evaluator,
                        &mut coset_evaluator,
                        theta,
                        advice_values,     // 拉格朗日基表示
                        &fixed_values,
                        instance_values,
                        advice_cosets,     // 扩展拉格朗日基表示
                        &fixed_cosets,
                        instance_cosets,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;
```

此操作将多个查表约束合并为一个，然后提交合并后的多项式。

这一操作可能涉及任何一列，包括 instance 列，advice 列和常数列（ fixed ），
因此他们都需要被传入 `lookup.commit_permuted` 函数。
参数 `value_evaluator` 和 `coset_evaluator` 用于延迟求值。

为了延迟求值，Halo2 将多项式表示为一种抽象语法树的叶结点，所有关于多项式的表达式均在此抽象语法树上计算。
而每个具体的多项式则保存在 `value_evaluator` 这样的求值器对象中。

`lookup.commit_permuted` 的具体操作如下

```rust
        // 利用挑战数theta合并多个带查询多项式的函数
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            // 即对于每一个查表约束，计算前述 A(X) 或者 S(X) 的拉格朗日基表示
            // 此处为抽象语法树表示
            let unpermuted_expressions: Vec<_> = expressions .iter()
                .map(|expression| { expression.evaluate(...) }) .collect();

            // 同上，但是扩展拉格朗日基表示
            let unpermuted_cosets: Vec<_> = expressions .iter()
                .map(|expression| { expression.evaluate(...) }) .collect();

            // 若有 A0(X), ..., An(X), 压缩为 theta^n An(X) + ... + theta A1(X) + A0(X)
            let compressed_expression = unpermuted_expressions.iter().fold(
                poly::Ast::ConstantTerm(C::Scalar::ZERO),
                |acc, expression| &(acc * *theta) + expression,
            );

            // 同上，但是扩展拉格朗日表示
            let compressed_coset = unpermuted_cosets.iter().fold(
                poly::Ast::<_, _, ExtendedLagrangeCoeff>::ConstantTerm(C::Scalar::ZERO),
                |acc, eval| acc * poly::Ast::ConstantTerm(*theta) + eval.clone(),
            );

            (
                // 这里仅仅返回了抽象语法树
                compressed_coset,
                // 这里求抽象语法树得到了具体的多项式
                value_evaluator.evaluate(&compressed_expression, domain),
            )
        };

        // Get values of input expressions involved in the lookup and compress them
        let (compressed_input_coset, compressed_input_expression) =
            compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let (compressed_table_coset, compressed_table_expression) =
            compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        // 这是 PLOOKUPS 协议要求的操作。得到多项式的拉格朗日系数是输入多项式的拉格朗日系数的一个置换。
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair::<C, _>(
            pk,
            params,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Closure to construct commitment to vector of values
        let mut commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>| {
            // 此处调用数论变换
            let poly = pk.vk.domain.lagrange_to_coeff(values.clone());
            let blind = Blind(C::Scalar::random(&mut rng));
            let commitment = params.commit_lagrange(values, blind).to_affine();
            (poly, blind, commitment)
        };

        // 提交置换后的多项式
        let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
            commit_values(&permuted_input_expression);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
            commit_values(&permuted_table_expression);

        // Hash permuted input commitment
        transcript.write_point(permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write_point(permuted_table_commitment)?;

        // 此处调用数论变换
        let permuted_input_coset = coset_evaluator
            .register_poly(pk.vk.domain.coeff_to_extended(permuted_input_poly.clone()));
        let permuted_table_coset = coset_evaluator
            .register_poly(pk.vk.domain.coeff_to_extended(permuted_table_poly.clone()));

        Ok(Permuted {
            compressed_input_expression, // 置换前、压缩后的输入多项式（即A(X)），拉格朗日基表示
            compressed_input_coset,      // 上一成员的扩展拉格朗日表示（未求值，仍为抽象语法树）
            permuted_input_expression,   // 置换后的输入多项式，拉格朗日表示
            permuted_input_poly,         // 系数表示
            permuted_input_coset,        // 扩展拉格朗日基表示
            permuted_input_blind,        // 盲化因子标量
            compressed_table_expression, // 同上，但是列表多项式
            compressed_table_coset,
            permuted_table_expression,
            permuted_table_poly,
            permuted_table_coset,
            permuted_table_blind,
        })
```

## 构造 PLOOKUPS 和置换关系的辅助多项式。

为了证明查表约束和置换约束成立，证明者需要先构造辅助多项式 \\(z_l(X)\\) 和 \\(z_p(X)\\)。

辅助多项式需要验证着发送挑战随机数 \\(beta\\) 和 \\(gamma\\):
```rust
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();alar();
```

如下的调用构造并提交辅助多项式。
```rust
    let permutations: Vec<permutation::prover::Committed<C, _>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_values,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut coset_evaluator,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let lookups: Vec<Vec<lookup::prover::Committed<C, _>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            // Construct and commit to products for each lookup
            lookups
                .into_iter()
                .map(|lookup| {
                    lookup.commit_product(
                        pk,
                        params,
                        beta,
                        gamma,
                        &mut coset_evaluator,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
```

下面是 `permutation.commit` 和 `lookup.commit_product` 调用的实现。

置换约束辅助多项式 \\(z_p(X)\\) 的构造在拉格朗日基下进行，然后转换到系数表示和扩展拉格朗日基表示。

```rust
        let domain = &pk.vk.domain;

        // How many columns can be included in a single permutation polynomial?
        // We need to multiply by z(X) and (1 - (l_last(X) + l_blind(X))). This
        // will never underflow because of the requirement of at least a degree
        // 3 circuit for the permutation argument.
        assert!(pk.vk.cs_degree >= 3);
        let chunk_len = pk.vk.cs_degree - 2;

        ...

        // 对于每一个置换约束...
        for (columns, permutations) in self
            .columns
            .chunks(chunk_len)
            .zip(pkey.permutations.chunks(chunk_len))
        {
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth column in this permutation,
            // and i is the ith row of the column.

            let mut modified_values = vec![C::Scalar::ONE; params.n as usize];
            // 填入 modified_values
            ...

            // The modified_values vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_values, for the jth column in
            // the permutation

            // Compute the evaluations of the permutation product polynomial
            // over our domain, starting with z[0] = 1
            let mut z = domain.lagrange_from_vec(z);
            // 设置盲化因子
            for z in &mut z[params.n as usize - blinding_factors..] {
                *z = C::Scalar::random(&mut rng);
            }
            // Set new last_z
            ...

            let blind = Blind(C::Scalar::random(&mut rng));
            let permutation_product_blind = blind;

            let permutation_product_commitment_projective = params.commit_lagrange(&z, blind);
            // 此处数论变换
            let z = domain.lagrange_to_coeff(z);
            let permutation_product_poly = z.clone();

            // 此处数论变换
            let permutation_product_coset =
                evaluator.register_poly(domain.coeff_to_extended(z.clone()));

            let permutation_product_commitment =
                permutation_product_commitment_projective.to_affine();

            // Hash the permutation product commitment
            transcript.write_point(permutation_product_commitment)?;

            sets.push(CommittedSet {
                // 辅助多项式的各种表示
                permutation_product_poly,
                permutation_product_coset,
                permutation_product_blind,
            });
        }

        Ok(Committed { sets })
```

查表约束辅助多项式的构造类似，也在拉格朗日基下进行。

```rust
let blinding_factors = pk.vk.cs.blinding_factors();
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.

        // Compute z
        ...

        let product_blind = Blind(C::Scalar::random(rng));
        let product_commitment = params.commit_lagrange(&z, product_blind).to_affine();
        let z = pk.vk.domain.lagrange_to_coeff(z);
        let product_coset = evaluator.register_poly(pk.vk.domain.coeff_to_extended(z.clone()));

        // Hash product commitment
        transcript.write_point(product_commitment)?;

        Ok(Committed::<C, _> {
            permuted: self,
            product_poly: z,
            product_coset,
            product_blind,
        })
```

## 生成消灭多项式 \\(h_{rand}(X)\\)

这仅仅是一个随机多项式。这个多项式确实被提交并打开，但是似乎和其他部分没有联系。

## 构造查表约束多项式和置换约束多项式

首先验证着发送挑战随机数
```rust
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();
```

然后构建约束多项式，约束多项式均使用扩展拉格朗日基表示。
下面 `permutation.cosntruct` 和 `p.construct` 调用均传入抽象语法树的叶结点，
返回构造好的抽象语法树。

```rust
    // Evaluate the h(X) polynomial's constraint system expressions for the permutation constraints.
    let (permutations, permutation_expressions): (Vec<_>, Vec<_>) = permutations
        .into_iter()
        .zip(advice_cosets.iter())
        .zip(instance_cosets.iter())
        .map(|((permutation, advice), instance)| {
            permutation.construct(
                pk,
                &pk.vk.cs.permutation,
                advice,
                &fixed_cosets,
                instance,
                &permutation_cosets,  // 置换多项式（电路相关），其中第i个多项式的第j个拉格朗日系数为 delta^i' omega^j'
                                      // 表示电路表格中(i, j)置换为(i', j')
                l0,                   // l_0, l_blind, l_last为一些拉格朗日基
                l_blind,
                l_last,
                beta,
                gamma,
            )
        })
        .unzip();

    let (lookups, lookup_expressions): (Vec<Vec<_>>, Vec<Vec<_>>) = lookups
        .into_iter()
        .map(|lookups| {
            // Evaluate the h(X) polynomial's constraint system expressions for the lookup constraints, if any.
            lookups
                .into_iter()
                .map(|p| p.construct(beta, gamma, l0, l_blind, l_last))
                .unzip()
        })
        .unzip();
```
具体构造的置换约束为

```rust
        let constructed = Constructed { /* 就是 self 里的数据 */ };

        // 这里将多个置换约束首尾相连
        let expressions = iter::empty()
            // Enforce only for the first set.
            // l_0(X) * (1 - z_0(X)) = 0
            .chain(...)
            // Enforce only for the last set.
            // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
            .chain(...)
            // Except for the first set, enforce.
            // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
            .chain(...)
            // And for all the sets we enforce:
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
            // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
            // )
            .chain( ...);
        (
            constructed,
            // 此处仍未求值，为抽象语法树
            expressions
        )
```

具体构造的查表约束为

```rust
        let expressions = iter::empty()
            // l_0(X) * (1 - z(X)) = 0
            .chain(...)
            // l_last(X) * (z(X)^2 - z(X)) = 0
            .chain(...)
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // ) = 0
            .chain(...)
            // Check that the first values in the permuted input expression and permuted
            // fixed expression are the same.
            // l_0(X) * (a'(X) - s'(X)) = 0
            .chain(...)
            // Check that each value in the permuted lookup input expression is either
            // equal to the value above it, or the value at the same index in the
            // permuted table expression.
            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            .chain(...);

        (
            Constructed { /* self 换个名字 */ },
            // 也是抽象语法树
            expressions,
        )
```

## 构造主约束多项式

集合所有约束多项式，然后构造消灭多项式 \\(h(X)\\)

```rust
    // 串联自定义约束多项式，和前面构造的置换约束多项式、查表约束多项式
    let expressions = advice_cosets
        .iter()
        .zip(instance_cosets.iter())
        .zip(permutation_expressions.into_iter())
        .zip(lookup_expressions.into_iter())
        .flat_map(
            |(((advice_cosets, instance_cosets), permutation_expressions), lookup_expressions)| {
                let fixed_cosets = &fixed_cosets;
                iter::empty()
                    // 自定义约束多项式
                    .chain(...)
                    // Permutation constraints, if any.
                    .chain(permutation_expressions.into_iter())
                    // Lookup constraints, if any.
                    .chain(lookup_expressions.into_iter().flatten())
            },
        );
    
    // 构造消灭多项式 h(X)
    let vanishing = vanishing.construct(
        params,
        domain,
        coset_evaluator,
        expressions,
        y,
        &mut rng,
        transcript,
    )?;
```

\\(h(X)\\) 构造如下

首先利用挑战随机数 \\(y\\) 合并各个约束多项式 \\(P_i(X)\\)
\\[P(X) = \sum y^i P_i(X)\\]

然后因为P(X)必然在\\(n\\)阶单位根\\(\omega\\)的幂处为0，
\\[h(X) = \frac{P(X)}{X^n - 1}\\]

是多项式。

最后分段提交 \\(h(X)\\)。

```rust
        // 求P(X)的具体值（在扩展拉格朗日基下）
        let h_poly = poly::Ast::distribute_powers(expressions, *y);
        let h_poly = evaluator.evaluate(&h_poly, domain);

        // 求h(X)
        let h_poly = domain.divide_by_vanishing_poly(h_poly);

        // 数论变换，转换为系数表示
        let h_poly = domain.extended_to_coeff(h_poly);

        // 分段提交
        ...
        for c in h_commitments.iter() {
            transcript.write_point(*c)?;
        }

        Ok(Constructed {
            h_pieces,
            h_blinds,
            committed: self,
        })
```

## 计算并打开各多项式

验证者发送挑战随机数 \\(x\\)
```rust
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
```

证明者在 \\(x\\) 处求值各多项式，包括 instance, advice, fixed 多项式、各辅助多项式和 \\(h(X)\\)，
然后将求值结果发送给验证者。接下来Halo2通过多点打开（multiopen）协议证明自己发送了正确的求值结果。

多点打开将要在多个点打开的若干多项式转换成在同一点打开的同一个多项式，然后由多项式提交协议证明自己发送了正确的求值结果。

- 多点打开协议见[Halo2 文档](https://zcash.github.io/halo2/design/proving-system/multipoint-opening.html)
- 多项式提交协议为 Bulletproofs 内积协议的一个变种
    - Bulletproofs 见 Bulletproofs: Short Proofs for Confidential Transactions and More
    - Halo2 变种见 Recursive Proof Composition without a Trusted Setup


